/**
 * Service Worker for GPX Analyzer PWA
 *
 * Caching strategies:
 * - Static assets: Cache First
 * - API endpoints: Network First with IndexedDB fallback
 * - Pages: Network First with cache fallback
 */

const CACHE_VERSION = 'v75';
const STATIC_CACHE = `gpx-static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `gpx-dynamic-${CACHE_VERSION}`;

// Static assets to pre-cache on install
const STATIC_ASSETS = [
  '/',
  '/saved',
  '/ride',
  '/static/css/main.css',
  '/static/css/ride.css',
  '/manifest.json',
  '/static/js/sw-register.js',
  '/static/js/offline-storage.js',
  '/static/js/offline-status.js',
  '/static/js/profile-renderer.js',
  '/static/js/router.js',
  '/static/icons/apple-touch-icon.png',
  '/static/icons/favicon.png',
  '/static/icons/logo-reality-check-my-route.png',
];

// Pages that should use cached version when offline (ignoring query params)
const OFFLINE_PAGES = ['/', '/saved', '/ride'];

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

  // Don't intercept POST requests - let them go to the server
  if (event.request.method !== 'GET') {
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
      url.pathname === '/climbs' ||
      url.pathname === '/api/detect-climbs') {
    event.respondWith(networkFirst(event.request, DYNAMIC_CACHE));
    return;
  }

  // Known pages - use network-first with offline fallback (handles both
  // top-level navigations and programmatic router fetch() requests)
  const pagePath = url.pathname;
  if (OFFLINE_PAGES.includes(pagePath)) {
    event.respondWith(networkFirstWithOfflineFallback(event.request, pagePath));
    return;
  }

  // Other navigations - network first
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

    // Try dynamic cache first
    let cachedResponse = await cache.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    // Also check static cache (for pre-cached pages like /saved)
    const staticCache = await caches.open(STATIC_CACHE);
    cachedResponse = await staticCache.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    // For navigation requests, try returning the home page
    if (request.mode === 'navigate') {
      // Check static cache for home page
      const homeResponse = await staticCache.match('/');
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

/**
 * Network-first with offline fallback for known pages
 * Tries network first (for fresh content), falls back to cache when offline
 */
async function networkFirstWithOfflineFallback(request, pagePath) {
  const staticCache = await caches.open(STATIC_CACHE);
  const dynamicCache = await caches.open(DYNAMIC_CACHE);

  try {
    // Try network first
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      // Cache the response for offline use
      dynamicCache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    // Network failed - we're offline, serve from cache
    console.log('[SW] Network failed, serving from cache:', pagePath);

    // Try dynamic cache first (has query params)
    let cachedResponse = await dynamicCache.match(request);
    if (cachedResponse) {
      return cachedResponse.clone();
    }

    // Try static cache with just the path
    cachedResponse = await staticCache.match(pagePath);
    if (cachedResponse) {
      return cachedResponse.clone();
    }

    // Try static cache with request
    cachedResponse = await staticCache.match(request);
    if (cachedResponse) {
      return cachedResponse.clone();
    }

    // Fallback to home page
    const homeResponse = await staticCache.match('/');
    if (homeResponse) {
      return homeResponse.clone();
    }

    return new Response('Offline - page not cached', {
      status: 503,
      headers: { 'Content-Type': 'text/plain' }
    });
  }
}

// Listen for messages from the main thread
self.addEventListener('message', (event) => {
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  if (event.data.type === 'GET_VERSION') {
    self.clients.matchAll().then(function(clients) {
      clients.forEach(function(client) {
        client.postMessage({ version: CACHE_VERSION });
      });
    });
  }
});
