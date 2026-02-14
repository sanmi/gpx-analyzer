/**
 * Offline Status Detection for GPX Analyzer PWA
 *
 * Monitors network connectivity and updates UI accordingly.
 */

(function(global) {
  'use strict';

  /**
   * OfflineStatus class - monitors and reports online/offline state
   */
  class OfflineStatus {
    constructor() {
      this.isOnline = navigator.onLine;
      this._listeners = [];
      this._banner = null;

      // Set up event listeners
      window.addEventListener('online', () => this._setOnline(true));
      window.addEventListener('offline', () => this._setOnline(false));

      // Initial UI update
      this._updateUI();
    }

    /**
     * Subscribe to online/offline changes
     */
    onStatusChange(callback) {
      this._listeners.push(callback);
      // Immediately notify of current state
      callback(this.isOnline);
    }

    /**
     * Check if currently online
     */
    get online() {
      return this.isOnline;
    }

    /**
     * Internal: update online state
     */
    _setOnline(status) {
      if (this.isOnline === status) return;

      this.isOnline = status;
      console.log('[OfflineStatus] Network status changed:', status ? 'online' : 'offline');

      this._updateUI();
      this._notifyListeners();
    }

    /**
     * Internal: update UI elements
     */
    _updateUI() {
      // Toggle body class
      document.body.classList.toggle('offline', !this.isOnline);

      // Show/hide offline banner
      if (!this.isOnline) {
        this._showBanner();
      } else {
        this._hideBanner();
      }

      // Update UI elements that depend on online status
      this._updateInteractiveElements();
    }

    /**
     * Internal: show offline banner
     */
    _showBanner() {
      if (this._banner) return;

      this._banner = document.createElement('div');
      this._banner.id = 'offline-banner';
      this._banner.className = 'offline-banner';
      this._banner.innerHTML = `
        <span>You're offline. Only saved routes are available.</span>
        <button type="button" class="offline-banner-link" onclick="if(typeof showOfflineSavedRoutes==='function')showOfflineSavedRoutes();" style="background:none;border:none;color:inherit;text-decoration:underline;cursor:pointer;font:inherit;">View Saved Routes</button>
      `;
      document.body.insertBefore(this._banner, document.body.firstChild);
    }

    /**
     * Internal: hide offline banner
     */
    _hideBanner() {
      if (this._banner) {
        this._banner.remove();
        this._banner = null;
      }
    }

    /**
     * Internal: update interactive elements based on online status
     */
    _updateInteractiveElements() {
      // Disable analyze button when offline (unless route is saved)
      const analyzeBtn = document.getElementById('analyze-btn') || document.querySelector('button[type="submit"]');
      if (analyzeBtn && !this.isOnline) {
        // We'll check if the current URL is saved before fully disabling
        // For now, just add a visual indicator
        analyzeBtn.classList.toggle('offline-disabled', !this.isOnline);
      }
    }

    /**
     * Internal: notify all listeners
     */
    _notifyListeners() {
      this._listeners.forEach(callback => {
        try {
          callback(this.isOnline);
        } catch (e) {
          console.error('[OfflineStatus] Listener error:', e);
        }
      });
    }
  }

  // Create singleton instance
  const offlineStatus = new OfflineStatus();

  // Export to global scope
  global.offlineStatus = offlineStatus;
  global.OfflineStatus = OfflineStatus;

})(typeof window !== 'undefined' ? window : this);
