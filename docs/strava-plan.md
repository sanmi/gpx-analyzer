# Strava Integration Plan

**Strava's API has a key limitation: you can only access your own activities, not other people's activities - even if they're public on the website.**

## Overview
Add support for importing routes and activities from Strava, in addition to existing RideWithGPS support. Use app-level OAuth token authentication (stored in config file, similar to RideWithGPS). No collections support initially.

## Authentication Approach

**App-level token** (not per-user OAuth):
- User registers a Strava API application at https://www.strava.com/settings/api
- Obtains `client_id`, `client_secret`, and generates a `refresh_token`
- Stores credentials in config file (`~/.config/gpx-analyzer/gpx-analyzer.json`)
- App uses refresh token to get short-lived access tokens as needed

```json
{
  "strava_client_id": "your-client-id",
  "strava_client_secret": "your-client-secret",
  "strava_refresh_token": "your-refresh-token"
}
```

Access tokens expire every 6 hours, so the app must:
1. Store current access token + expiration time in memory/cache
2. Check expiration before each request
3. Use refresh token to get new access token when expired

## URL Patterns

| Type | URL Pattern | Example |
|------|-------------|---------|
| Route | `strava.com/routes/{id}` | `https://www.strava.com/routes/123456` |
| Activity | `strava.com/activities/{id}` | `https://www.strava.com/activities/789012` |

## API Endpoints

### Routes (simpler - GPX export available)
```
GET https://www.strava.com/api/v3/routes/{id}/export_gpx
Authorization: Bearer {access_token}
```
Returns: GPX XML file (can use existing GPX parser)

### Activities (more complex - no GPX export)
```
# Get activity metadata
GET https://www.strava.com/api/v3/activities/{id}
Authorization: Bearer {access_token}

# Get time-series data streams
GET https://www.strava.com/api/v3/activities/{id}/streams?keys=latlng,altitude,time,distance,velocity_smooth,watts,heartrate,cadence&key_type=distance
Authorization: Bearer {access_token}
```

Available streams: `time`, `latlng`, `distance`, `altitude`, `velocity_smooth`, `heartrate`, `cadence`, `watts`, `temp`, `moving`, `grade_smooth`

## Data Mapping

### Strava Route → TrackPoint
GPX export provides standard GPX format - use existing `parse_gpx()` function.

**Note**: Strava does NOT provide surface type data (R/S values like RideWithGPS). All points will have default CRR, `unpaved=False`.

### Strava Activity → TripPoint
Map streams data to TripPoint structure:

| Strava Stream | TripPoint Field |
|---------------|-----------------|
| `latlng[0]` | `lat` |
| `latlng[1]` | `lon` |
| `altitude` | `elevation` |
| `distance` | `distance` |
| `velocity_smooth` | `speed` (already m/s) |
| `time` | `timestamp` (seconds from start → convert to unix) |
| `watts` | `power` |
| `heartrate` | `heart_rate` |
| `cadence` | `cadence` |

Activity metadata provides: `name`, `distance`, `total_elevation_gain`, `moving_time`, `elapsed_time`, `average_speed`, `average_watts`

## Implementation

### New File: `src/gpx_analyzer/strava.py`

```python
# URL patterns
STRAVA_ROUTE_PATTERN = r"^https?://(?:www\.)?strava\.com/routes/(\d+)"
STRAVA_ACTIVITY_PATTERN = r"^https?://(?:www\.)?strava\.com/activities/(\d+)"

# Detection functions
def is_strava_route_url(url: str) -> bool
def is_strava_activity_url(url: str) -> bool
def extract_strava_route_id(url: str) -> str
def extract_strava_activity_id(url: str) -> str

# Authentication
def _load_strava_config() -> dict
def _get_access_token() -> str  # Handles token refresh

# Data fetching
def get_strava_route(url: str, baseline_crr: float) -> tuple[list[TrackPoint], dict]
def get_strava_activity(url: str) -> tuple[list[TripPoint], dict]

# Caching (similar to RideWithGPS)
# - Route GPX cache: ~/.cache/gpx-analyzer/strava_routes/
# - Activity cache: ~/.cache/gpx-analyzer/strava_activities/
```

### Modify: `src/gpx_analyzer/web.py`

1. Import strava module
2. Update URL detection logic to check Strava patterns
3. Route to appropriate fetcher based on URL type
4. Handle case where surface data is unavailable (no gravel overlay for Strava)

### Modify: `src/gpx_analyzer/cli.py`

1. Add Strava URL support to main analysis flow
2. Support `--compare-trip` with Strava activity URLs

### Modify: `README.md`

1. Add Strava API credentials setup instructions
2. Document supported Strava URL formats
3. Note limitations (no surface data)

## Key Differences from RideWithGPS

| Feature | RideWithGPS | Strava |
|---------|-------------|--------|
| Auth | API key + auth token | OAuth2 refresh token |
| Route GPX | Direct download | Direct download |
| Activity/Trip GPX | Direct download | NOT available - use Streams API |
| Surface data | R/S values per point | Not available |
| Collections | Supported | Not supported (initially) |

## Caching Strategy

Similar to RideWithGPS:
- **Route GPX**: LRU cache in `~/.cache/gpx-analyzer/strava_routes/`
- **Activity JSON**: LRU cache in `~/.cache/gpx-analyzer/strava_activities/`
- **Access token**: In-memory with expiration tracking

No ETag support needed initially (Strava data changes less frequently than RideWithGPS route edits).

## Rate Limiting

Strava limits: 200 requests/15min, 2000/day

Mitigation:
- Aggressive caching (already planned)
- No collection support means fewer bulk requests
- Log rate limit headers for monitoring

## Files to Modify

1. **New**: `src/gpx_analyzer/strava.py` - All Strava-specific code
2. **Modify**: `src/gpx_analyzer/web.py` - URL routing, disable gravel overlay for Strava
3. **Modify**: `src/gpx_analyzer/cli.py` - URL detection and routing
4. **Modify**: `README.md` - Documentation
5. **New**: `tests/unit/test_strava.py` - Unit tests

## Verification

1. Set up Strava API credentials in config file
2. Test route analysis: `gpx-analyzer https://www.strava.com/routes/123456`
3. Test activity analysis: `gpx-analyzer https://www.strava.com/activities/789012`
4. Test web UI with Strava URLs
5. Test comparison mode: RideWithGPS route vs Strava activity
6. Verify caching works (second request should be faster)
7. Verify token refresh works (wait 6+ hours or manually expire token)

## Out of Scope (Future)

- Strava collections/clubs
- Per-user OAuth (private activities)
- Segment analysis
- Surface data estimation from Strava
