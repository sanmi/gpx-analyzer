/**
 * Service Worker for GPX Analyzer PWA
 *
 * Caching strategies:
 * - Static assets: Cache First
 * - API endpoints: Network First with IndexedDB fallback
 * - Pages: Network First with cache fallback
 */

const CACHE_VERSION = 'v3';
const STATIC_CACHE = `gpx-static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `gpx-dynamic-${CACHE_VERSION}`;

// Static assets to pre-cache on install
const STATIC_ASSETS = [
  '/',
  '/static/css/main.css',
  '/static/css/ride.css',
  '/static/manifest.json',
];

// Install event - pre-cache static assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker...');
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('[SW] Pre-caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker...');
  event.waitUntil(
    caches.keys()
      .then(keys => {
        return Promise.all(
          keys
            .filter(key => key !== STATIC_CACHE && key !== DYNAMIC_CACHE)
            .map(key => {
              console.log('[SW] Deleting old cache:', key);
              return caches.delete(key);
            })
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Only handle same-origin requests
  if (url.origin !== location.origin) {
    return;
  }

  // Static assets - cache first
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(cacheFirst(event.request, STATIC_CACHE));
    return;
  }

  // API endpoints - network first
  if (url.pathname === '/elevation-profile-data' ||
      url.pathname === '/analyze' ||
      url.pathname === '/climbs') {
    event.respondWith(networkFirst(event.request, DYNAMIC_CACHE));
    return;
  }

  // Pages - network first with fallback
  if (event.request.mode === 'navigate') {
    event.respondWith(networkFirst(event.request, DYNAMIC_CACHE));
    return;
  }

  // Default - network first
  event.respondWith(networkFirst(event.request, DYNAMIC_CACHE));
});

/**
 * Cache-first strategy: Try cache, fall back to network
 */
async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(request);

  if (cachedResponse) {
    return cachedResponse;
  }

  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('[SW] Cache-first fetch failed:', error);
    // Return a basic offline page if available
    return caches.match('/');
  }
}

/**
 * Network-first strategy: Try network, fall back to cache
 */
async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);

  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('[SW] Network request failed, trying cache:', request.url);
    const cachedResponse = await cache.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    // For navigation requests, try returning the home page
    if (request.mode === 'navigate') {
      const homeResponse = await cache.match('/');
      if (homeResponse) {
        return homeResponse;
      }
    }

    // Return an offline error response
    return new Response(
      JSON.stringify({ error: 'Offline and no cached data available' }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

// Listen for messages from the main thread
self.addEventListener('message', (event) => {
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
