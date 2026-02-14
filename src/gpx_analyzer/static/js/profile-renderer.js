/**
 * Canvas-based Elevation Profile Renderer
 *
 * Replaces server-rendered PNG profiles with client-side Canvas rendering.
 * Enables offline zoom and interaction without network requests.
 */

(function(global) {
  'use strict';

  // Grade bins (matches Python GRADE_BINS)
  const GRADE_BINS = [-Infinity, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, Infinity];

  // Main grade colors for bins < 10% (matches Python MAIN_GRADE_COLORS)
  const MAIN_GRADE_COLORS = [
    '#4a90d9', '#5a9fd9', '#6aaee0', '#7abde7', '#8acbef', '#9adaf6',  // downhill blues
    '#cccccc',  // flat gray
    '#ffb399', '#ff9966', '#ff7f33', '#ff6600', '#e55a00'  // uphill oranges
  ];

  // Steep grade colors for >= 10% (matches Python STEEP_GRADE_COLORS)
  const STEEP_GRADE_COLORS = ['#e55a00', '#cc4400', '#b33300', '#992200', '#801100', '#660000'];

  /**
   * Map grade percentage to color (matches Python grade_to_color exactly)
   */
  function gradeToColor(grade) {
    if (grade === null || grade === undefined) {
      return '#ffffff';  // White for stopped segments
    }

    // For grades >= 10%, use steep colors
    if (grade >= 10) {
      if (grade < 12) return STEEP_GRADE_COLORS[0];
      if (grade < 14) return STEEP_GRADE_COLORS[1];
      if (grade < 16) return STEEP_GRADE_COLORS[2];
      if (grade < 18) return STEEP_GRADE_COLORS[3];
      if (grade < 20) return STEEP_GRADE_COLORS[4];
      return STEEP_GRADE_COLORS[5];
    }

    // For grades < 10%, use main colors based on bins
    for (let i = 1; i < GRADE_BINS.length; i++) {
      if (grade < GRADE_BINS[i]) {
        return MAIN_GRADE_COLORS[i - 1];
      }
    }
    return MAIN_GRADE_COLORS[MAIN_GRADE_COLORS.length - 1];
  }

  /**
   * Format time value for axis labels
   */
  function formatTimeLabel(hours) {
    if (hours < 1) {
      return Math.round(hours * 60) + 'm';
    }
    const h = Math.floor(hours);
    const m = Math.round((hours - h) * 60);
    if (m === 0) {
      return h + 'h';
    }
    return h + 'h' + (m < 10 ? '0' : '') + m;
  }

  /**
   * Format elevation value for axis labels
   */
  function formatElevLabel(meters, imperial) {
    if (imperial) {
      return Math.round(meters * 3.28084) + ' ft';
    }
    return Math.round(meters) + ' m';
  }

  /**
   * ElevationProfileRenderer class
   *
   * Renders elevation profiles on a Canvas element with grade-based coloring,
   * zoom support, and interactive features.
   */
  class ElevationProfileRenderer {
    constructor(canvas, data, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.data = data;  // { times, elevations, grades, speeds, distances, powers, works, ... }

      // Options
      this.imperial = options.imperial || false;
      this.showSpeedOverlay = options.showSpeedOverlay || false;
      this.showGradeOverlay = options.showGradeOverlay || false;
      this.showGravel = options.showGravel || false;

      // View state (normalized 0-1 range)
      this.viewRange = { start: 0, end: 1 };

      // Selection state (normalized 0-1 range, null if no selection)
      this.selection = null;

      // Margins for axes (will be calculated based on canvas size)
      this.margins = { top: 20, right: 60, bottom: 30, left: 50 };

      // Hover position (null if not hovering)
      this.hoverX = null;

      // Device pixel ratio for high-DPI displays
      this.dpr = window.devicePixelRatio || 1;

      // Tunnel and unpaved ranges (for highlighting)
      this.tunnelRanges = data.tunnel_ranges || [];
      this.unpavedRanges = data.unpaved_ranges || [];

      // Climb regions (for ride.html)
      this.climbRegions = data.climb_regions || [];

      // Bind event handlers
      this._onMouseMove = this._onMouseMove.bind(this);
      this._onMouseLeave = this._onMouseLeave.bind(this);
      this._onMouseDown = this._onMouseDown.bind(this);
      this._onMouseUp = this._onMouseUp.bind(this);
      this._onTouchStart = this._onTouchStart.bind(this);
      this._onTouchMove = this._onTouchMove.bind(this);
      this._onTouchEnd = this._onTouchEnd.bind(this);

      // Selection drag state
      this._isDragging = false;
      this._dragStart = null;

      // Callbacks
      this.onHover = null;  // Called with data point info on hover
      this.onSelect = null;  // Called with selection range info
    }

    /**
     * Initialize the renderer and set up event listeners
     */
    init() {
      this._setupCanvas();
      this._attachEventListeners();
      this.render();
    }

    /**
     * Set up canvas for high-DPI rendering
     */
    _setupCanvas() {
      const rect = this.canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;

      // Scale canvas for high-DPI
      this.canvas.width = width * this.dpr;
      this.canvas.height = height * this.dpr;
      this.ctx.scale(this.dpr, this.dpr);

      // Store CSS dimensions
      this.width = width;
      this.height = height;

      // Calculate plot area
      this.plotLeft = this.margins.left;
      this.plotRight = this.width - this.margins.right;
      this.plotTop = this.margins.top;
      this.plotBottom = this.height - this.margins.bottom;
      this.plotWidth = this.plotRight - this.plotLeft;
      this.plotHeight = this.plotBottom - this.plotTop;
    }

    /**
     * Attach mouse/touch event listeners
     */
    _attachEventListeners() {
      this.canvas.addEventListener('mousemove', this._onMouseMove);
      this.canvas.addEventListener('mouseleave', this._onMouseLeave);
      this.canvas.addEventListener('mousedown', this._onMouseDown);
      this.canvas.addEventListener('mouseup', this._onMouseUp);
      this.canvas.addEventListener('touchstart', this._onTouchStart, { passive: false });
      this.canvas.addEventListener('touchmove', this._onTouchMove, { passive: false });
      this.canvas.addEventListener('touchend', this._onTouchEnd);
    }

    /**
     * Remove event listeners (for cleanup)
     */
    destroy() {
      this.canvas.removeEventListener('mousemove', this._onMouseMove);
      this.canvas.removeEventListener('mouseleave', this._onMouseLeave);
      this.canvas.removeEventListener('mousedown', this._onMouseDown);
      this.canvas.removeEventListener('mouseup', this._onMouseUp);
      this.canvas.removeEventListener('touchstart', this._onTouchStart);
      this.canvas.removeEventListener('touchmove', this._onTouchMove);
      this.canvas.removeEventListener('touchend', this._onTouchEnd);
    }

    /**
     * Get data indices for current view range
     */
    _getViewIndices() {
      const n = this.data.times.length;
      const startIdx = Math.floor(this.viewRange.start * (n - 1));
      const endIdx = Math.ceil(this.viewRange.end * (n - 1));
      return { startIdx, endIdx };
    }

    /**
     * Convert normalized position (0-1) to canvas X coordinate
     */
    _normToCanvasX(norm) {
      // Adjust for current view range
      const viewNorm = (norm - this.viewRange.start) / (this.viewRange.end - this.viewRange.start);
      return this.plotLeft + viewNorm * this.plotWidth;
    }

    /**
     * Convert canvas X coordinate to normalized position (0-1)
     */
    _canvasXToNorm(x) {
      const viewNorm = (x - this.plotLeft) / this.plotWidth;
      return this.viewRange.start + viewNorm * (this.viewRange.end - this.viewRange.start);
    }

    /**
     * Convert elevation to canvas Y coordinate
     */
    _elevToCanvasY(elev, minElev, maxElev) {
      const range = maxElev - minElev;
      const normalized = (elev - minElev) / range;
      return this.plotBottom - normalized * this.plotHeight;
    }

    /**
     * Main render function
     */
    render() {
      const ctx = this.ctx;
      const { times, elevations, grades } = this.data;
      const n = times.length;

      if (n < 2) return;

      // Clear canvas
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, this.width, this.height);

      // Get view range indices
      const { startIdx, endIdx } = this._getViewIndices();

      // Calculate elevation range for visible data
      let minElev = Infinity, maxElev = -Infinity;
      for (let i = startIdx; i <= endIdx && i < n; i++) {
        if (elevations[i] < minElev) minElev = elevations[i];
        if (elevations[i] > maxElev) maxElev = elevations[i];
      }

      // Add padding to elevation range
      const elevRange = maxElev - minElev;
      minElev = Math.max(0, minElev - elevRange * 0.05);
      maxElev = maxElev + elevRange * 0.1;

      // Get time range for visible data
      const minTime = times[startIdx];
      const maxTime = times[Math.min(endIdx, n - 1)];

      // Draw tunnel/anomaly highlights (behind everything)
      this._drawTunnelHighlights(minTime, maxTime, maxElev);

      // Draw unpaved highlights if enabled
      if (this.showGravel) {
        this._drawUnpavedHighlights(minTime, maxTime, maxElev);
      }

      // Draw climb regions if present
      this._drawClimbRegions(minTime, maxTime, maxElev);

      // Draw grade-colored elevation fill
      this._drawElevationFill(startIdx, endIdx, minElev, maxElev, minTime, maxTime);

      // Draw elevation outline
      this._drawElevationLine(startIdx, endIdx, minElev, maxElev, minTime, maxTime);

      // Draw axes
      this._drawAxes(minTime, maxTime, minElev, maxElev);

      // Draw speed or grade overlay if enabled
      if (this.showSpeedOverlay) {
        this._drawSpeedOverlay(startIdx, endIdx, minTime, maxTime);
      } else if (this.showGradeOverlay) {
        this._drawGradeOverlay(startIdx, endIdx, minTime, maxTime);
      }

      // Draw selection highlight
      if (this.selection) {
        this._drawSelection();
      }

      // Draw hover cursor
      if (this.hoverX !== null) {
        this._drawHoverCursor();
      }
    }

    /**
     * Draw grade-colored elevation fill
     */
    _drawElevationFill(startIdx, endIdx, minElev, maxElev, minTime, maxTime) {
      const ctx = this.ctx;
      const { times, elevations, grades } = this.data;
      const n = times.length;
      const timeRange = maxTime - minTime;

      for (let i = startIdx; i < endIdx && i < n - 1; i++) {
        const t0 = times[i];
        const t1 = times[i + 1];
        const e0 = elevations[i];
        const e1 = elevations[i + 1];
        const grade = grades[i];

        // Convert to canvas coordinates
        const x0 = this.plotLeft + ((t0 - minTime) / timeRange) * this.plotWidth;
        const x1 = this.plotLeft + ((t1 - minTime) / timeRange) * this.plotWidth;
        const y0 = this._elevToCanvasY(e0, minElev, maxElev);
        const y1 = this._elevToCanvasY(e1, minElev, maxElev);
        const yBottom = this.plotBottom;

        // Draw polygon
        ctx.fillStyle = gradeToColor(grade);
        ctx.beginPath();
        ctx.moveTo(x0, yBottom);
        ctx.lineTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1, yBottom);
        ctx.closePath();
        ctx.fill();
      }
    }

    /**
     * Draw elevation outline
     */
    _drawElevationLine(startIdx, endIdx, minElev, maxElev, minTime, maxTime) {
      const ctx = this.ctx;
      const { times, elevations } = this.data;
      const n = times.length;
      const timeRange = maxTime - minTime;

      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 0.5;
      ctx.beginPath();

      for (let i = startIdx; i <= endIdx && i < n; i++) {
        const t = times[i];
        const e = elevations[i];
        const x = this.plotLeft + ((t - minTime) / timeRange) * this.plotWidth;
        const y = this._elevToCanvasY(e, minElev, maxElev);

        if (i === startIdx) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
    }

    /**
     * Draw axes with labels
     */
    _drawAxes(minTime, maxTime, minElev, maxElev) {
      const ctx = this.ctx;

      // Axis lines
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 1;

      // X-axis
      ctx.beginPath();
      ctx.moveTo(this.plotLeft, this.plotBottom);
      ctx.lineTo(this.plotRight, this.plotBottom);
      ctx.stroke();

      // Y-axis
      ctx.beginPath();
      ctx.moveTo(this.plotLeft, this.plotTop);
      ctx.lineTo(this.plotLeft, this.plotBottom);
      ctx.stroke();

      // X-axis labels (time)
      ctx.fillStyle = '#333333';
      ctx.font = '11px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';

      const timeRange = maxTime - minTime;
      const numXTicks = Math.min(6, Math.floor(this.plotWidth / 60));
      for (let i = 0; i <= numXTicks; i++) {
        const t = minTime + (i / numXTicks) * timeRange;
        const x = this.plotLeft + (i / numXTicks) * this.plotWidth;
        ctx.fillText(formatTimeLabel(t), x, this.plotBottom + 5);

        // Tick mark
        ctx.beginPath();
        ctx.moveTo(x, this.plotBottom);
        ctx.lineTo(x, this.plotBottom + 3);
        ctx.stroke();
      }

      // Y-axis labels (elevation)
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';

      const elevRange = maxElev - minElev;
      const numYTicks = Math.min(5, Math.floor(this.plotHeight / 40));
      for (let i = 0; i <= numYTicks; i++) {
        const e = minElev + (i / numYTicks) * elevRange;
        const y = this._elevToCanvasY(e, minElev, maxElev);
        ctx.fillText(formatElevLabel(e, this.imperial), this.plotLeft - 5, y);

        // Grid line
        ctx.strokeStyle = '#e0e0e0';
        ctx.beginPath();
        ctx.moveTo(this.plotLeft, y);
        ctx.lineTo(this.plotRight, y);
        ctx.stroke();
        ctx.strokeStyle = '#333333';
      }
    }

    /**
     * Draw tunnel/anomaly highlights
     */
    _drawTunnelHighlights(minTime, maxTime, maxElev) {
      if (!this.tunnelRanges || this.tunnelRanges.length === 0) return;

      const ctx = this.ctx;
      const timeRange = maxTime - minTime;

      ctx.fillStyle = 'rgba(128, 128, 128, 0.15)';

      for (const [startTime, endTime] of this.tunnelRanges) {
        // Check if range is visible
        if (endTime < minTime || startTime > maxTime) continue;

        const x0 = this.plotLeft + Math.max(0, (startTime - minTime) / timeRange) * this.plotWidth;
        const x1 = this.plotLeft + Math.min(1, (endTime - minTime) / timeRange) * this.plotWidth;

        ctx.fillRect(x0, this.plotTop, x1 - x0, this.plotHeight);
      }
    }

    /**
     * Draw unpaved section highlights
     */
    _drawUnpavedHighlights(minTime, maxTime, maxElev) {
      if (!this.unpavedRanges || this.unpavedRanges.length === 0) return;

      const ctx = this.ctx;
      const timeRange = maxTime - minTime;

      // Draw brown hatching for unpaved sections
      ctx.fillStyle = 'rgba(139, 90, 43, 0.2)';

      for (const [startTime, endTime] of this.unpavedRanges) {
        if (endTime < minTime || startTime > maxTime) continue;

        const x0 = this.plotLeft + Math.max(0, (startTime - minTime) / timeRange) * this.plotWidth;
        const x1 = this.plotLeft + Math.min(1, (endTime - minTime) / timeRange) * this.plotWidth;

        ctx.fillRect(x0, this.plotTop, x1 - x0, this.plotHeight);
      }
    }

    /**
     * Draw climb region highlights
     */
    _drawClimbRegions(minTime, maxTime, maxElev) {
      if (!this.climbRegions || this.climbRegions.length === 0) return;

      const ctx = this.ctx;
      const timeRange = maxTime - minTime;

      ctx.fillStyle = 'rgba(0, 180, 0, 0.15)';

      for (const climb of this.climbRegions) {
        const startTime = climb.start_time;
        const endTime = climb.end_time;

        if (endTime < minTime || startTime > maxTime) continue;

        const x0 = this.plotLeft + Math.max(0, (startTime - minTime) / timeRange) * this.plotWidth;
        const x1 = this.plotLeft + Math.min(1, (endTime - minTime) / timeRange) * this.plotWidth;

        ctx.fillRect(x0, this.plotTop, x1 - x0, this.plotHeight);
      }
    }

    /**
     * Draw speed overlay
     */
    _drawSpeedOverlay(startIdx, endIdx, minTime, maxTime) {
      if (!this.data.speeds || this.data.speeds.length === 0) return;

      const ctx = this.ctx;
      const { times, speeds } = this.data;
      const n = times.length;
      const timeRange = maxTime - minTime;

      // Calculate speed range
      let maxSpeed = 0;
      for (let i = startIdx; i <= endIdx && i < n; i++) {
        if (speeds[i] > maxSpeed) maxSpeed = speeds[i];
      }
      maxSpeed = Math.ceil(maxSpeed / 10) * 10;  // Round up to nearest 10

      // Draw speed line
      ctx.strokeStyle = '#0066cc';
      ctx.lineWidth = 1.2;
      ctx.beginPath();

      for (let i = startIdx; i <= endIdx && i < n; i++) {
        const t = times[i];
        const s = speeds[i];
        const x = this.plotLeft + ((t - minTime) / timeRange) * this.plotWidth;
        const y = this.plotBottom - (s / maxSpeed) * this.plotHeight;

        if (i === startIdx) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();

      // Draw right Y-axis for speed
      ctx.strokeStyle = '#0066cc';
      ctx.beginPath();
      ctx.moveTo(this.plotRight, this.plotTop);
      ctx.lineTo(this.plotRight, this.plotBottom);
      ctx.stroke();

      // Speed axis labels
      ctx.fillStyle = '#0066cc';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';

      const unit = this.imperial ? 'mph' : 'km/h';
      const speedFactor = this.imperial ? 0.621371 : 1;

      for (let i = 0; i <= 4; i++) {
        const s = (i / 4) * maxSpeed;
        const y = this.plotBottom - (i / 4) * this.plotHeight;
        const displaySpeed = Math.round(s * speedFactor);
        ctx.fillText(displaySpeed + (i === 4 ? ' ' + unit : ''), this.plotRight + 5, y);
      }
    }

    /**
     * Draw grade overlay
     */
    _drawGradeOverlay(startIdx, endIdx, minTime, maxTime) {
      const ctx = this.ctx;
      const { times, grades } = this.data;
      const n = times.length;
      const timeRange = maxTime - minTime;

      // Calculate grade range
      let minGrade = 0, maxGrade = 0;
      for (let i = startIdx; i <= endIdx && i < n; i++) {
        if (grades[i] !== null) {
          if (grades[i] < minGrade) minGrade = grades[i];
          if (grades[i] > maxGrade) maxGrade = grades[i];
        }
      }

      // Round to nearest 5
      minGrade = Math.floor(minGrade / 5) * 5;
      maxGrade = Math.ceil(maxGrade / 5) * 5;
      const gradeRange = maxGrade - minGrade;

      // Draw grade line
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 1.2;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();

      let started = false;
      for (let i = startIdx; i <= endIdx && i < n; i++) {
        if (grades[i] === null) continue;

        const t = times[i];
        const g = grades[i];
        const x = this.plotLeft + ((t - minTime) / timeRange) * this.plotWidth;
        const y = this.plotBottom - ((g - minGrade) / gradeRange) * this.plotHeight;

        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
      ctx.globalAlpha = 1;

      // Draw right Y-axis for grade
      ctx.strokeStyle = '#333333';
      ctx.beginPath();
      ctx.moveTo(this.plotRight, this.plotTop);
      ctx.lineTo(this.plotRight, this.plotBottom);
      ctx.stroke();

      // Grade axis labels
      ctx.fillStyle = '#333333';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';

      const numTicks = 4;
      for (let i = 0; i <= numTicks; i++) {
        const g = minGrade + (i / numTicks) * gradeRange;
        const y = this.plotBottom - (i / numTicks) * this.plotHeight;
        ctx.fillText(Math.round(g) + (i === numTicks ? '%' : ''), this.plotRight + 5, y);
      }
    }

    /**
     * Draw selection highlight
     */
    _drawSelection() {
      if (!this.selection) return;

      const ctx = this.ctx;
      const x0 = this._normToCanvasX(this.selection.start);
      const x1 = this._normToCanvasX(this.selection.end);

      // Selection highlight
      ctx.fillStyle = 'rgba(255, 107, 53, 0.3)';
      ctx.fillRect(x0, this.plotTop, x1 - x0, this.plotHeight);

      // Selection borders
      ctx.strokeStyle = '#ff6b35';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x0, this.plotTop);
      ctx.lineTo(x0, this.plotBottom);
      ctx.moveTo(x1, this.plotTop);
      ctx.lineTo(x1, this.plotBottom);
      ctx.stroke();
    }

    /**
     * Draw hover cursor
     */
    _drawHoverCursor() {
      if (this.hoverX === null) return;

      const ctx = this.ctx;

      ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(this.hoverX, this.plotTop);
      ctx.lineTo(this.hoverX, this.plotBottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    /**
     * Get data point at canvas X coordinate
     */
    getDataAtX(canvasX) {
      const norm = this._canvasXToNorm(canvasX);
      if (norm < 0 || norm > 1) return null;

      const { times, elevations, grades, speeds, distances, powers, works } = this.data;
      const n = times.length;
      const idx = Math.round(norm * (n - 1));

      if (idx < 0 || idx >= n) return null;

      return {
        index: idx,
        time: times[idx],
        elevation: elevations[idx],
        grade: grades[idx],
        speed: speeds ? speeds[idx] : null,
        distance: distances ? distances[idx] : null,
        power: powers ? powers[idx] : null,
        work: works ? works[idx] : null
      };
    }

    /**
     * Calculate statistics for a selection range
     */
    getSelectionStats(startNorm, endNorm) {
      const { times, elevations, grades, speeds, distances, powers, works, elev_gains, elev_losses } = this.data;
      const n = times.length;

      const startIdx = Math.round(startNorm * (n - 1));
      const endIdx = Math.round(endNorm * (n - 1));

      if (startIdx >= endIdx) return null;

      let totalDistance = 0;
      let totalTime = times[endIdx] - times[startIdx];
      let totalElevGain = 0;
      let totalElevLoss = 0;
      let totalWork = 0;
      let avgSpeed = 0;
      let speedCount = 0;

      for (let i = startIdx; i < endIdx; i++) {
        if (distances && distances[i]) totalDistance += distances[i];
        if (elev_gains && elev_gains[i]) totalElevGain += elev_gains[i];
        if (elev_losses && elev_losses[i]) totalElevLoss += elev_losses[i];
        if (works && works[i]) totalWork += works[i];
        if (speeds && speeds[i] !== null) {
          avgSpeed += speeds[i];
          speedCount++;
        }
      }

      return {
        startTime: times[startIdx],
        endTime: times[endIdx],
        duration: totalTime,
        distance: totalDistance,
        elevGain: totalElevGain,
        elevLoss: totalElevLoss,
        work: totalWork / 1000,  // Convert J to kJ
        avgSpeed: speedCount > 0 ? avgSpeed / speedCount : 0
      };
    }

    // --- Zoom methods ---

    /**
     * Zoom to a specific range
     */
    zoom(startNorm, endNorm) {
      this.viewRange.start = Math.max(0, startNorm);
      this.viewRange.end = Math.min(1, endNorm);
      this.render();
    }

    /**
     * Reset zoom to full view
     */
    resetZoom() {
      this.viewRange = { start: 0, end: 1 };
      this.selection = null;
      this.render();
    }

    /**
     * Zoom to current selection
     */
    zoomToSelection() {
      if (!this.selection) return;
      this.zoom(this.selection.start, this.selection.end);
      this.selection = null;
    }

    // --- Event handlers ---

    _onMouseMove(e) {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Check if in plot area
      if (x < this.plotLeft || x > this.plotRight || y < this.plotTop || y > this.plotBottom) {
        this.hoverX = null;
        this.render();
        return;
      }

      this.hoverX = x;

      // Handle drag selection
      if (this._isDragging) {
        const startNorm = this._canvasXToNorm(this._dragStart);
        const endNorm = this._canvasXToNorm(x);
        this.selection = {
          start: Math.min(startNorm, endNorm),
          end: Math.max(startNorm, endNorm)
        };
      }

      this.render();

      // Fire hover callback
      if (this.onHover) {
        const data = this.getDataAtX(x);
        if (data) {
          this.onHover(data, x, y);
        }
      }
    }

    _onMouseLeave(e) {
      this.hoverX = null;
      this._isDragging = false;
      this.render();
    }

    _onMouseDown(e) {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;

      if (x >= this.plotLeft && x <= this.plotRight) {
        this._isDragging = true;
        this._dragStart = x;
        this.selection = null;
      }
    }

    _onMouseUp(e) {
      if (this._isDragging && this.selection) {
        // Fire selection callback
        if (this.onSelect) {
          const stats = this.getSelectionStats(this.selection.start, this.selection.end);
          if (stats) {
            this.onSelect(stats, this.selection);
          }
        }
      }
      this._isDragging = false;
    }

    _onTouchStart(e) {
      if (e.touches.length === 1) {
        e.preventDefault();
        const rect = this.canvas.getBoundingClientRect();
        const x = e.touches[0].clientX - rect.left;

        if (x >= this.plotLeft && x <= this.plotRight) {
          this._isDragging = true;
          this._dragStart = x;
          this.selection = null;
        }
      }
    }

    _onTouchMove(e) {
      if (e.touches.length === 1 && this._isDragging) {
        e.preventDefault();
        const rect = this.canvas.getBoundingClientRect();
        const x = e.touches[0].clientX - rect.left;

        const startNorm = this._canvasXToNorm(this._dragStart);
        const endNorm = this._canvasXToNorm(x);
        this.selection = {
          start: Math.min(startNorm, endNorm),
          end: Math.max(startNorm, endNorm)
        };

        this.render();
      }
    }

    _onTouchEnd(e) {
      if (this._isDragging && this.selection) {
        if (this.onSelect) {
          const stats = this.getSelectionStats(this.selection.start, this.selection.end);
          if (stats) {
            this.onSelect(stats, this.selection);
          }
        }
      }
      this._isDragging = false;
    }

    // --- Resize handling ---

    /**
     * Handle canvas resize
     */
    resize() {
      this._setupCanvas();
      this.render();
    }
  }

  // Export to global scope
  global.ElevationProfileRenderer = ElevationProfileRenderer;
  global.gradeToColor = gradeToColor;

})(typeof window !== 'undefined' ? window : this);
