# Progressive Web App (PWA) Implementation Plan

## Overview

This document outlines the plan to convert the GPX Analyzer web application into a Progressive Web App with full offline support. The goal is to allow users to download routes/trips beforehand and then view analyses, zoom in/out of elevation profiles, and interact with all features while offline.

## Current Architecture Summary

| Component | Current State | PWA Impact |
|-----------|---------------|------------|
| Frontend | Vanilla JS inline in HTML templates | Extract to external files for caching |
| Elevation Profiles | Server-rendered PNG images | Move to client-side Canvas rendering |
| Profile Data | JSON API (`/elevation-profile-data`) | Cache in IndexedDB |
| Route Analysis | Server-side Python physics | Pre-compute and cache results |
| Settings | localStorage | Already PWA-compatible |
| Styling | External CSS files | Service Worker cacheable |

## Key Design Decisions

### 1. Client-Side Elevation Profile Rendering

**Why:** Server-rendered PNG images cannot support offline zoom without caching every possible zoom level (impractical).

**Solution:** Render elevation profiles using HTML Canvas on the client side using cached JSON profile data.

**Benefits:**
- Infinite zoom levels from single data download
- Smaller data transfer (JSON vs multiple PNGs)
- Faster interactions (no network round-trip)
- True offline support

### 2. Pre-Download Model

Users will explicitly "save for offline" routes/trips which will:
1. Fetch and cache all analysis data
2. Fetch and cache profile data points
3. Store in IndexedDB for persistence
4. Mark as available offline in UI

---

## Implementation Phases

### Phase 1: Foundation (Extract & Restructure)

**Goal:** Prepare codebase for PWA without changing functionality.

#### 1.1 Extract JavaScript to External Files

Create modular JavaScript files from inline code:

```
src/gpx_analyzer/static/js/
├── app.js                 # Main application initialization
├── elevation-profile.js   # Profile interactions (from setupElevationProfile)
├── storage.js             # localStorage/IndexedDB abstraction
├── api.js                 # API fetch functions
├── utils.js               # Shared utilities
└── sw-register.js         # Service Worker registration
```

**Tasks:**
- [ ] Extract `setupElevationProfile()` function (~750 lines) to `elevation-profile.js`
- [ ] Extract API calls to `api.js`
- [ ] Extract localStorage helpers to `storage.js`
- [ ] Create `app.js` for initialization
- [ ] Update templates to use external scripts
- [ ] Add module bundling (optional: use ES modules directly)

#### 1.2 Add Web App Manifest

Create `manifest.json`:

```json
{
  "name": "GPX Analyzer - Reality Check my Route",
  "short_name": "GPX Analyzer",
  "description": "Analyze cycling routes with physics-based time estimates",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#f5f5f7",
  "theme_color": "#FF6B35",
  "icons": [
    {
      "src": "/static/icons/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/static/icons/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

**Tasks:**
- [ ] Create manifest.json
- [ ] Create app icons (192x192, 512x512)
- [ ] Add manifest link to HTML templates
- [ ] Add meta tags for iOS Safari

#### 1.3 Create Service Worker Shell

Basic service worker for static asset caching:

```javascript
// sw.js
const CACHE_NAME = 'gpx-analyzer-v1';
const STATIC_ASSETS = [
  '/',
  '/static/css/main.css',
  '/static/css/ride.css',
  '/static/js/app.js',
  '/static/js/elevation-profile.js',
  // ... other static assets
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
```

**Tasks:**
- [ ] Create `sw.js` service worker
- [ ] Create `sw-register.js` for registration
- [ ] Implement cache-first strategy for static assets
- [ ] Add version management for cache updates

---

### Phase 2: Client-Side Profile Rendering

**Goal:** Replace server-rendered PNG profiles with Canvas-based client rendering.

#### 2.1 Create Canvas Profile Renderer

New file: `static/js/profile-renderer.js`

```javascript
class ElevationProfileRenderer {
  constructor(canvas, data) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = data;  // { times, elevations, grades, speeds, ... }
    this.viewRange = { start: 0, end: 1 };  // Normalized 0-1
  }

  // Grade to color mapping (match existing Python colors)
  gradeToColor(grade) {
    const mainColors = [
      '#4a90d9', '#5a9fd9', '#6aaee0', '#7abde7', '#8acbef', '#9adaf6',
      '#cccccc',
      '#ffb399', '#ff9966', '#ff7f33', '#ff6600', '#e55a00'
    ];
    const steepColors = ['#e55a00', '#cc4400', '#b33300', '#992200', '#801100', '#660000'];
    // ... implement grade binning logic
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);

    // Calculate visible data range
    const startIdx = Math.floor(this.viewRange.start * this.data.times.length);
    const endIdx = Math.ceil(this.viewRange.end * this.data.times.length);

    // Draw grade-colored elevation profile
    this.drawElevationFill(startIdx, endIdx);
    this.drawElevationLine(startIdx, endIdx);
    this.drawAxes();
    this.drawOverlay();  // Speed or grade overlay
  }

  zoom(startNorm, endNorm) {
    this.viewRange = { start: startNorm, end: endNorm };
    this.render();
  }

  resetZoom() {
    this.viewRange = { start: 0, end: 1 };
    this.render();
  }
}
```

**Tasks:**
- [ ] Implement `ElevationProfileRenderer` class
- [ ] Port grade-to-color mapping from Python
- [ ] Implement elevation fill with grade colors
- [ ] Implement elevation line drawing
- [ ] Implement axis labels and grid
- [ ] Implement speed/grade overlay rendering
- [ ] Implement zoom functionality
- [ ] Implement tooltip rendering
- [ ] Implement selection highlighting
- [ ] Match visual appearance of current Matplotlib output

#### 2.2 Update Profile Data API

Ensure `/elevation-profile-data` returns all needed fields:

```python
@app.route("/elevation-profile-data")
def elevation_profile_data():
    # Return full data for client-side rendering
    return jsonify({
        "times": times_hours,
        "elevations": elevations,
        "grades": grades,
        "speeds": speeds_kmh,
        "distances": distances,
        "powers": powers,
        "works": works,
        "elev_gains": elev_gains,
        "elev_losses": elev_losses,
        "total_time": total_time,
        "route_name": route_name,
        "tunnel_ranges": tunnel_time_ranges,  # For anomaly highlighting
        "unpaved_ranges": unpaved_time_ranges,  # For gravel highlighting
    })
```

**Tasks:**
- [ ] Add tunnel_ranges to API response
- [ ] Add unpaved_ranges to API response
- [ ] Ensure consistent data format
- [ ] Add climb regions to API response (for ride.html)

#### 2.3 Integrate Canvas Renderer

Update templates to use Canvas instead of `<img>`:

```html
<!-- Before -->
<img id="elevation-profile" src="/elevation-profile?...">

<!-- After -->
<canvas id="elevation-profile-canvas"></canvas>
<script>
  const canvas = document.getElementById('elevation-profile-canvas');
  const renderer = new ElevationProfileRenderer(canvas, profileData);
  renderer.render();
</script>
```

**Tasks:**
- [ ] Replace `<img>` with `<canvas>` in templates
- [ ] Initialize renderer with fetched profile data
- [ ] Wire up existing interaction handlers to new renderer
- [ ] Ensure responsive canvas sizing
- [ ] Handle high-DPI displays (devicePixelRatio)

---

### Phase 3: IndexedDB Storage

**Goal:** Implement persistent offline storage for routes and analyses.

#### 3.1 Design Database Schema

```javascript
// Database: gpx-analyzer-db
// Version: 1

const DB_SCHEMA = {
  // Saved routes/trips for offline access
  savedRoutes: {
    keyPath: 'url',
    indexes: ['savedAt', 'name'],
    // { url, name, savedAt, analysisParams }
  },

  // Cached analysis results
  analyses: {
    keyPath: 'cacheKey',
    indexes: ['url', 'cachedAt'],
    // { cacheKey, url, params, result, cachedAt }
  },

  // Cached profile data
  profileData: {
    keyPath: 'cacheKey',
    indexes: ['url', 'cachedAt'],
    // { cacheKey, url, data, cachedAt }
  },

  // Cached climb detection results
  climbs: {
    keyPath: 'cacheKey',
    indexes: ['url', 'cachedAt'],
    // { cacheKey, url, sensitivity, climbs, cachedAt }
  }
};
```

#### 3.2 Implement Storage Layer

New file: `static/js/offline-storage.js`

```javascript
class OfflineStorage {
  constructor() {
    this.dbName = 'gpx-analyzer-db';
    this.version = 1;
    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        // Create object stores...
      };
    });
  }

  async saveRoute(url, name, analysisParams) { /* ... */ }
  async getAnalysis(cacheKey) { /* ... */ }
  async saveAnalysis(cacheKey, url, params, result) { /* ... */ }
  async getProfileData(cacheKey) { /* ... */ }
  async saveProfileData(cacheKey, url, data) { /* ... */ }
  async getSavedRoutes() { /* ... */ }
  async deleteRoute(url) { /* ... */ }
  async getStorageStats() { /* ... */ }
}
```

**Tasks:**
- [ ] Implement IndexedDB wrapper class
- [ ] Implement CRUD operations for each store
- [ ] Implement cache key generation (match Python logic)
- [ ] Add storage quota management
- [ ] Add data expiration/cleanup logic

#### 3.3 Implement "Save for Offline" Feature

UI components:

```html
<!-- Save button on analysis results -->
<button id="save-offline" class="save-offline-btn">
  <span class="icon">📥</span>
  Save for Offline
</button>

<!-- Saved routes list -->
<div class="saved-routes-section">
  <h3>Saved Routes</h3>
  <ul id="saved-routes-list">
    <!-- Dynamically populated -->
  </ul>
</div>
```

**Tasks:**
- [ ] Add "Save for Offline" button to analysis results
- [ ] Implement save workflow (fetch all data, store in IndexedDB)
- [ ] Show progress indicator during save
- [ ] Add "Saved Routes" section to homepage
- [ ] Implement delete/manage saved routes
- [ ] Show storage usage stats
- [ ] Add visual indicator for offline-available routes

---

### Phase 4: Service Worker Enhancements

**Goal:** Implement intelligent caching strategies for offline support.

#### 4.1 Caching Strategies

```javascript
// sw.js - Enhanced service worker

// Strategy: Cache First (static assets)
const STATIC_CACHE = 'gpx-static-v1';

// Strategy: Network First, Cache Fallback (API data)
const API_CACHE = 'gpx-api-v1';

// Strategy: Stale While Revalidate (profile data)
const PROFILE_CACHE = 'gpx-profiles-v1';

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Static assets - cache first
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(cacheFirst(event.request, STATIC_CACHE));
    return;
  }

  // Profile data API - check IndexedDB first, then network
  if (url.pathname === '/elevation-profile-data') {
    event.respondWith(handleProfileDataRequest(event.request));
    return;
  }

  // Analysis API - check IndexedDB first
  if (url.pathname === '/analyze' || url.pathname === '/') {
    event.respondWith(handleAnalysisRequest(event.request));
    return;
  }

  // Default - network first
  event.respondWith(networkFirst(event.request, API_CACHE));
});
```

**Tasks:**
- [ ] Implement cache-first strategy for static assets
- [ ] Implement network-first for API requests
- [ ] Implement IndexedDB lookup for offline data
- [ ] Add background sync for queued analyses
- [ ] Implement cache versioning and cleanup

#### 4.2 Offline Detection & UI

```javascript
// offline-status.js
class OfflineStatus {
  constructor() {
    this.isOnline = navigator.onLine;
    this.listeners = [];

    window.addEventListener('online', () => this.setOnline(true));
    window.addEventListener('offline', () => this.setOnline(false));
  }

  setOnline(status) {
    this.isOnline = status;
    this.notifyListeners();
    this.updateUI();
  }

  updateUI() {
    document.body.classList.toggle('offline', !this.isOnline);
    const banner = document.getElementById('offline-banner');
    if (banner) {
      banner.hidden = this.isOnline;
    }
  }
}
```

**Tasks:**
- [ ] Add offline status detection
- [ ] Add offline banner/indicator
- [ ] Disable unavailable features when offline
- [ ] Show cached-only routes when offline
- [ ] Queue analysis requests for when online

---

### Phase 5: Polish & Optimization

**Goal:** Optimize performance and user experience.

#### 5.1 Performance Optimizations

- [ ] Lazy load JavaScript modules
- [ ] Implement virtual scrolling for large route lists
- [ ] Optimize Canvas rendering (requestAnimationFrame)
- [ ] Compress cached data (consider LZ-string)
- [ ] Implement data pagination for large profiles

#### 5.2 iOS-Specific Enhancements

```html
<!-- iOS meta tags -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="GPX Analyzer">
<link rel="apple-touch-icon" href="/static/icons/apple-touch-icon.png">

<!-- Splash screens -->
<link rel="apple-touch-startup-image" href="/static/splash/splash-640x1136.png"
      media="(device-width: 320px)">
<!-- ... more splash screens for different devices -->
```

**Tasks:**
- [ ] Add iOS meta tags
- [ ] Create Apple touch icons
- [ ] Create splash screens for various iOS devices
- [ ] Test and fix iOS Safari-specific issues
- [ ] Handle iOS storage limitations gracefully

#### 5.3 User Experience

- [ ] Add install prompt for PWA
- [ ] Implement onboarding for offline features
- [ ] Add sync status indicators
- [ ] Implement data export/import
- [ ] Add storage management UI

---

## File Structure After Implementation

```
src/gpx_analyzer/
├── static/
│   ├── css/
│   │   ├── main.css
│   │   ├── ride.css
│   │   └── offline.css          # NEW: Offline-specific styles
│   ├── js/
│   │   ├── app.js               # NEW: Main app initialization
│   │   ├── api.js               # NEW: API fetch functions
│   │   ├── elevation-profile.js # NEW: Profile interactions
│   │   ├── profile-renderer.js  # NEW: Canvas rendering
│   │   ├── offline-storage.js   # NEW: IndexedDB wrapper
│   │   ├── offline-status.js    # NEW: Online/offline detection
│   │   ├── storage.js           # NEW: localStorage helpers
│   │   ├── utils.js             # NEW: Shared utilities
│   │   └── sw-register.js       # NEW: Service worker registration
│   ├── icons/
│   │   ├── icon-192.png         # NEW: PWA icon
│   │   ├── icon-512.png         # NEW: PWA icon
│   │   └── apple-touch-icon.png # NEW: iOS icon
│   └── splash/                  # NEW: iOS splash screens
├── templates/
│   ├── index.html               # Modified: External JS, Canvas
│   └── ride.html                # Modified: External JS, Canvas
├── sw.js                        # NEW: Service worker
├── manifest.json                # NEW: Web app manifest
└── web.py                       # Modified: New API endpoints
```

---

## Testing Plan

### Unit Tests

- [ ] Profile renderer draws correctly at various zoom levels
- [ ] IndexedDB storage operations work correctly
- [ ] Cache key generation matches Python implementation
- [ ] Grade-to-color mapping matches Python implementation

### Integration Tests

- [ ] Full offline workflow (save → disconnect → view)
- [ ] Zoom interactions work with Canvas renderer
- [ ] Selection statistics match server-side calculations
- [ ] Data survives browser restart

### Device Testing

- [ ] iPhone Safari (primary target)
- [ ] iPad Safari
- [ ] Android Chrome
- [ ] Desktop Chrome/Firefox/Safari

### Offline Testing

- [ ] Chrome DevTools offline mode
- [ ] Airplane mode on real devices
- [ ] Slow network simulation
- [ ] Storage quota exceeded scenarios

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Foundation | 2-3 days | None |
| Phase 2: Canvas Rendering | 3-5 days | Phase 1 |
| Phase 3: IndexedDB Storage | 2-3 days | Phase 1 |
| Phase 4: Service Worker | 2-3 days | Phases 1-3 |
| Phase 5: Polish | 2-3 days | Phases 1-4 |
| **Total** | **11-17 days** | |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| iOS Safari IndexedDB bugs | Data loss | Implement fallback to localStorage for critical data |
| Storage quota limits (~50MB) | Can't save many routes | Implement data compression, show quota warnings |
| Canvas rendering performance | Slow on older devices | Implement level-of-detail rendering, limit data points |
| Service Worker complexity | Stale data issues | Clear versioning strategy, force update mechanism |
| Matplotlib visual differences | Users notice change | Carefully match colors, line weights, fonts |

---

## Success Criteria

1. **Offline Viewing**: Users can view saved routes/trips with full interactivity while offline
2. **Zoom Support**: Elevation profile zoom works offline with smooth performance
3. **Data Persistence**: Saved data survives browser restarts and app updates
4. **Install Experience**: Users can "Add to Home Screen" with proper icon and splash
5. **Performance**: Canvas rendering performs as well as or better than image loading
6. **Storage Efficiency**: Typical route requires <500KB of storage

---

## Future Enhancements (Post-PWA)

1. **Background Sync**: Queue analysis requests while offline, sync when online
2. **Push Notifications**: Notify when queued analyses complete
3. **Share Target**: Accept shared URLs from RideWithGPS app
4. **Capacitor Wrapper**: Convert to native app for App Store distribution
5. **Offline Route Planning**: Edit routes while offline, sync changes later
