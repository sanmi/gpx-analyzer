/**
 * IndexedDB Storage for GPX Analyzer PWA
 *
 * Provides persistent storage for offline access to routes and trips.
 */

(function(global) {
  'use strict';

  const DB_NAME = 'gpx-analyzer-db';
  const DB_VERSION = 1;

  // Object store names
  const STORES = {
    SAVED_ROUTES: 'savedRoutes',
    ANALYSES: 'analyses',
    PROFILE_DATA: 'profileData',
    CLIMBS: 'climbs'
  };

  /**
   * OfflineStorage class - manages IndexedDB operations
   */
  class OfflineStorage {
    constructor() {
      this.db = null;
      this._initPromise = null;
    }

    /**
     * Initialize the database connection
     */
    async init() {
      if (this.db) return this.db;
      if (this._initPromise) return this._initPromise;

      this._initPromise = new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => {
          console.error('[OfflineStorage] Failed to open database:', request.error);
          reject(request.error);
        };

        request.onsuccess = () => {
          this.db = request.result;
          console.log('[OfflineStorage] Database opened successfully');
          resolve(this.db);
        };

        request.onupgradeneeded = (event) => {
          const db = event.target.result;
          console.log('[OfflineStorage] Upgrading database schema...');

          // Saved routes - explicitly saved by user for offline access
          if (!db.objectStoreNames.contains(STORES.SAVED_ROUTES)) {
            const savedRoutes = db.createObjectStore(STORES.SAVED_ROUTES, { keyPath: 'url' });
            savedRoutes.createIndex('savedAt', 'savedAt', { unique: false });
            savedRoutes.createIndex('name', 'name', { unique: false });
          }

          // Cached analysis results
          if (!db.objectStoreNames.contains(STORES.ANALYSES)) {
            const analyses = db.createObjectStore(STORES.ANALYSES, { keyPath: 'cacheKey' });
            analyses.createIndex('url', 'url', { unique: false });
            analyses.createIndex('cachedAt', 'cachedAt', { unique: false });
          }

          // Cached profile data (for elevation charts)
          if (!db.objectStoreNames.contains(STORES.PROFILE_DATA)) {
            const profileData = db.createObjectStore(STORES.PROFILE_DATA, { keyPath: 'cacheKey' });
            profileData.createIndex('url', 'url', { unique: false });
            profileData.createIndex('cachedAt', 'cachedAt', { unique: false });
          }

          // Cached climb detection results
          if (!db.objectStoreNames.contains(STORES.CLIMBS)) {
            const climbs = db.createObjectStore(STORES.CLIMBS, { keyPath: 'cacheKey' });
            climbs.createIndex('url', 'url', { unique: false });
            climbs.createIndex('cachedAt', 'cachedAt', { unique: false });
          }
        };
      });

      return this._initPromise;
    }

    /**
     * Check if a route/trip is saved for offline use
     */
    async isSaved(url) {
      await this.init();
      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction([STORES.SAVED_ROUTES], 'readonly');
        const store = transaction.objectStore(STORES.SAVED_ROUTES);
        const request = store.get(url);

        request.onsuccess = () => resolve(!!request.result);
        request.onerror = () => reject(request.error);
      });
    }

    /**
     * Save a route/trip for offline use
     */
    async saveRoute(url, name, analysisParams, analysisResult, profileData, climbs = null) {
      await this.init();
      const now = Date.now();

      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction(
          [STORES.SAVED_ROUTES, STORES.ANALYSES, STORES.PROFILE_DATA, STORES.CLIMBS],
          'readwrite'
        );

        transaction.onerror = () => reject(transaction.error);
        transaction.oncomplete = () => {
          console.log('[OfflineStorage] Route saved for offline:', url);
          resolve();
        };

        // Save route metadata
        const savedRoutes = transaction.objectStore(STORES.SAVED_ROUTES);
        savedRoutes.put({
          url: url,
          name: name,
          savedAt: now,
          analysisParams: analysisParams,
          sizeKb: this._estimateSize(analysisResult, profileData, climbs)
        });

        // Save analysis result
        const analyses = transaction.objectStore(STORES.ANALYSES);
        const analysisCacheKey = this._makeAnalysisCacheKey(url, analysisParams);
        analyses.put({
          cacheKey: analysisCacheKey,
          url: url,
          params: analysisParams,
          result: analysisResult,
          cachedAt: now
        });

        // Save profile data
        const profiles = transaction.objectStore(STORES.PROFILE_DATA);
        const profileCacheKey = this._makeProfileCacheKey(url, analysisParams);
        profiles.put({
          cacheKey: profileCacheKey,
          url: url,
          data: profileData,
          cachedAt: now
        });

        // Save climbs if provided
        if (climbs) {
          const climbsStore = transaction.objectStore(STORES.CLIMBS);
          const climbCacheKey = this._makeClimbCacheKey(url, analysisParams);
          climbsStore.put({
            cacheKey: climbCacheKey,
            url: url,
            climbs: climbs,
            cachedAt: now
          });
        }
      });
    }

    /**
     * Delete a saved route
     */
    async deleteRoute(url) {
      await this.init();

      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction(
          [STORES.SAVED_ROUTES, STORES.ANALYSES, STORES.PROFILE_DATA, STORES.CLIMBS],
          'readwrite'
        );

        transaction.onerror = () => reject(transaction.error);
        transaction.oncomplete = () => {
          console.log('[OfflineStorage] Route deleted:', url);
          resolve();
        };

        // Delete from saved routes
        transaction.objectStore(STORES.SAVED_ROUTES).delete(url);

        // Delete associated analyses
        const analyses = transaction.objectStore(STORES.ANALYSES);
        const analysisIndex = analyses.index('url');
        analysisIndex.openCursor(IDBKeyRange.only(url)).onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            cursor.continue();
          }
        };

        // Delete associated profile data
        const profiles = transaction.objectStore(STORES.PROFILE_DATA);
        const profileIndex = profiles.index('url');
        profileIndex.openCursor(IDBKeyRange.only(url)).onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            cursor.continue();
          }
        };

        // Delete associated climbs
        const climbsStore = transaction.objectStore(STORES.CLIMBS);
        const climbIndex = climbsStore.index('url');
        climbIndex.openCursor(IDBKeyRange.only(url)).onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            cursor.delete();
            cursor.continue();
          }
        };
      });
    }

    /**
     * Get all saved routes
     */
    async getSavedRoutes() {
      await this.init();

      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction([STORES.SAVED_ROUTES], 'readonly');
        const store = transaction.objectStore(STORES.SAVED_ROUTES);
        const request = store.getAll();

        request.onsuccess = () => {
          // Sort by savedAt descending (newest first)
          const routes = request.result.sort((a, b) => b.savedAt - a.savedAt);
          resolve(routes);
        };
        request.onerror = () => reject(request.error);
      });
    }

    /**
     * Get cached analysis for a URL
     */
    async getAnalysis(url, params) {
      await this.init();
      const cacheKey = this._makeAnalysisCacheKey(url, params);

      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction([STORES.ANALYSES], 'readonly');
        const store = transaction.objectStore(STORES.ANALYSES);
        const request = store.get(cacheKey);

        request.onsuccess = () => {
          const entry = request.result;
          resolve(entry ? entry.result : null);
        };
        request.onerror = () => reject(request.error);
      });
    }

    /**
     * Get cached profile data for a URL
     */
    async getProfileData(url, params) {
      await this.init();
      const cacheKey = this._makeProfileCacheKey(url, params);

      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction([STORES.PROFILE_DATA], 'readonly');
        const store = transaction.objectStore(STORES.PROFILE_DATA);
        const request = store.get(cacheKey);

        request.onsuccess = () => {
          const entry = request.result;
          resolve(entry ? entry.data : null);
        };
        request.onerror = () => reject(request.error);
      });
    }

    /**
     * Get cached climbs for a URL
     */
    async getClimbs(url, params) {
      await this.init();
      const cacheKey = this._makeClimbCacheKey(url, params);

      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction([STORES.CLIMBS], 'readonly');
        const store = transaction.objectStore(STORES.CLIMBS);
        const request = store.get(cacheKey);

        request.onsuccess = () => {
          const entry = request.result;
          resolve(entry ? entry.climbs : null);
        };
        request.onerror = () => reject(request.error);
      });
    }

    /**
     * Get storage statistics
     */
    async getStorageStats() {
      await this.init();

      const routes = await this.getSavedRoutes();
      const totalSizeKb = routes.reduce((sum, r) => sum + (r.sizeKb || 0), 0);

      // Get browser quota if available
      let quotaKb = 50 * 1024; // Default 50MB estimate
      let usedKb = totalSizeKb;

      if (navigator.storage && navigator.storage.estimate) {
        try {
          const estimate = await navigator.storage.estimate();
          quotaKb = Math.round(estimate.quota / 1024);
          usedKb = Math.round(estimate.usage / 1024);
        } catch (e) {
          console.warn('[OfflineStorage] Could not get storage estimate:', e);
        }
      }

      return {
        savedRoutes: routes.length,
        totalSizeKb: totalSizeKb,
        quotaKb: quotaKb,
        usedKb: usedKb,
        usagePercent: Math.round((usedKb / quotaKb) * 100)
      };
    }

    /**
     * Clear all offline data
     */
    async clearAll() {
      await this.init();

      return new Promise((resolve, reject) => {
        const transaction = this.db.transaction(
          [STORES.SAVED_ROUTES, STORES.ANALYSES, STORES.PROFILE_DATA, STORES.CLIMBS],
          'readwrite'
        );

        transaction.onerror = () => reject(transaction.error);
        transaction.oncomplete = () => {
          console.log('[OfflineStorage] All data cleared');
          resolve();
        };

        transaction.objectStore(STORES.SAVED_ROUTES).clear();
        transaction.objectStore(STORES.ANALYSES).clear();
        transaction.objectStore(STORES.PROFILE_DATA).clear();
        transaction.objectStore(STORES.CLIMBS).clear();
      });
    }

    // --- Private helper methods ---

    _makeAnalysisCacheKey(url, params) {
      // Match Python cache key format
      const parts = [
        url,
        params.climbing_power,
        params.flat_power,
        params.mass,
        params.headwind,
        params.descent_braking_factor || 1.0,
        params.descending_power || 20,
        params.gravel_power_factor || 0.9,
        params.smoothing || 50,
        params.smoothing_override || false
      ];
      return this._hashString(parts.join('|'));
    }

    _makeProfileCacheKey(url, params) {
      // Similar to Python make_profile_cache_key
      const parts = [
        url,
        params.climbing_power,
        params.flat_power,
        params.descending_power || 20,
        params.mass,
        params.headwind,
        params.descent_braking_factor || 1.0
      ];
      return 'profile_' + this._hashString(parts.join('|'));
    }

    _makeClimbCacheKey(url, params) {
      return `climb|${url}|${params.sensitivity || 50}`;
    }

    _hashString(str) {
      // Simple hash function (not cryptographic)
      let hash = 0;
      for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
      }
      return Math.abs(hash).toString(16);
    }

    _estimateSize(analysisResult, profileData, climbs) {
      // Rough estimate of data size in KB
      let size = 0;
      if (analysisResult) {
        size += JSON.stringify(analysisResult).length / 1024;
      }
      if (profileData) {
        size += JSON.stringify(profileData).length / 1024;
      }
      if (climbs) {
        size += JSON.stringify(climbs).length / 1024;
      }
      return Math.round(size);
    }
  }

  // Create singleton instance
  const offlineStorage = new OfflineStorage();

  // Export to global scope
  global.offlineStorage = offlineStorage;
  global.OfflineStorage = OfflineStorage;

})(typeof window !== 'undefined' ? window : this);
