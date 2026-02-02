import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpx_analyzer import ridewithgps


class TestIsRidewithgpsUrl:
    def test_valid_url(self):
        assert ridewithgps.is_ridewithgps_url("https://ridewithgps.com/routes/53835626")

    def test_valid_url_with_www(self):
        assert ridewithgps.is_ridewithgps_url(
            "https://www.ridewithgps.com/routes/53835626"
        )

    def test_valid_url_http(self):
        assert ridewithgps.is_ridewithgps_url("http://ridewithgps.com/routes/53835626")

    def test_local_file_path(self):
        assert not ridewithgps.is_ridewithgps_url("/path/to/file.gpx")

    def test_other_url(self):
        assert not ridewithgps.is_ridewithgps_url("https://strava.com/routes/123")

    def test_empty_string(self):
        assert not ridewithgps.is_ridewithgps_url("")

    def test_invalid_route_path(self):
        assert not ridewithgps.is_ridewithgps_url("https://ridewithgps.com/trips/123")

    def test_valid_url_with_privacy_code(self):
        assert ridewithgps.is_ridewithgps_url(
            "https://ridewithgps.com/routes/53835626?privacy_code=ABC123"
        )


class TestExtractRouteId:
    def test_valid_url(self):
        assert (
            ridewithgps.extract_route_id("https://ridewithgps.com/routes/53835626")
            == 53835626
        )

    def test_valid_url_with_www(self):
        assert (
            ridewithgps.extract_route_id("https://www.ridewithgps.com/routes/12345")
            == 12345
        )

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid RideWithGPS URL"):
            ridewithgps.extract_route_id("https://example.com/routes/123")

    def test_local_path_raises(self):
        with pytest.raises(ValueError, match="Invalid RideWithGPS URL"):
            ridewithgps.extract_route_id("/path/to/file.gpx")


class TestExtractPrivacyCode:
    def test_url_with_privacy_code(self):
        url = "https://ridewithgps.com/routes/53835626?privacy_code=ABC123"
        assert ridewithgps.extract_privacy_code(url) == "ABC123"

    def test_url_without_privacy_code(self):
        url = "https://ridewithgps.com/routes/53835626"
        assert ridewithgps.extract_privacy_code(url) is None

    def test_url_with_other_params(self):
        url = "https://ridewithgps.com/routes/53835626?foo=bar&privacy_code=XYZ789&baz=qux"
        assert ridewithgps.extract_privacy_code(url) == "XYZ789"

    def test_url_with_empty_privacy_code(self):
        url = "https://ridewithgps.com/routes/53835626?privacy_code="
        # Empty privacy_code is treated as None (not useful)
        assert ridewithgps.extract_privacy_code(url) is None


class TestCacheOperations:
    @pytest.fixture
    def temp_cache_dir(self, tmp_path, monkeypatch):
        """Set up a temporary cache directory for testing."""
        cache_dir = tmp_path / ".cache" / "gpx-analyzer"
        routes_dir = cache_dir / "routes"
        routes_dir.mkdir(parents=True)
        index_path = cache_dir / "cache_index.json"

        monkeypatch.setattr(ridewithgps, "CACHE_DIR", cache_dir)
        monkeypatch.setattr(ridewithgps, "ROUTES_DIR", routes_dir)
        monkeypatch.setattr(ridewithgps, "CACHE_INDEX_PATH", index_path)

        return {
            "cache_dir": cache_dir,
            "routes_dir": routes_dir,
            "index_path": index_path,
        }

    def test_get_cached_path_exists(self, temp_cache_dir):
        routes_dir = temp_cache_dir["routes_dir"]
        gpx_file = routes_dir / "12345.gpx"
        gpx_file.write_text("<gpx></gpx>")

        result = ridewithgps._get_cached_path(12345)
        assert result == gpx_file

    def test_get_cached_path_not_exists(self, temp_cache_dir):
        result = ridewithgps._get_cached_path(99999)
        assert result is None

    def test_save_to_cache(self, temp_cache_dir):
        routes_dir = temp_cache_dir["routes_dir"]
        index_path = temp_cache_dir["index_path"]

        gpx_data = b"<gpx>test data</gpx>"
        path = ridewithgps._save_to_cache(12345, gpx_data)

        assert path == routes_dir / "12345.gpx"
        assert path.read_bytes() == gpx_data
        assert index_path.exists()

        with index_path.open() as f:
            index = json.load(f)
        assert "12345" in index

    def test_update_lru(self, temp_cache_dir):
        index_path = temp_cache_dir["index_path"]

        ridewithgps._update_lru(12345)

        with index_path.open() as f:
            index = json.load(f)
        assert "12345" in index
        assert isinstance(index["12345"], float)

    def test_update_lru_updates_existing(self, temp_cache_dir):
        index_path = temp_cache_dir["index_path"]

        ridewithgps._update_lru(12345)
        with index_path.open() as f:
            first_time = json.load(f)["12345"]

        ridewithgps._update_lru(12345)
        with index_path.open() as f:
            second_time = json.load(f)["12345"]

        assert second_time >= first_time


class TestLruEviction:
    @pytest.fixture
    def temp_cache_with_files(self, tmp_path, monkeypatch):
        """Set up a cache with existing files for LRU testing."""
        cache_dir = tmp_path / ".cache" / "gpx-analyzer"
        routes_dir = cache_dir / "routes"
        routes_dir.mkdir(parents=True)
        index_path = cache_dir / "cache_index.json"

        monkeypatch.setattr(ridewithgps, "CACHE_DIR", cache_dir)
        monkeypatch.setattr(ridewithgps, "ROUTES_DIR", routes_dir)
        monkeypatch.setattr(ridewithgps, "CACHE_INDEX_PATH", index_path)

        return {
            "cache_dir": cache_dir,
            "routes_dir": routes_dir,
            "index_path": index_path,
        }

    def test_enforce_lru_limit_under_limit(self, temp_cache_with_files):
        routes_dir = temp_cache_with_files["routes_dir"]
        index_path = temp_cache_with_files["index_path"]

        index = {}
        for i in range(5):
            (routes_dir / f"{i}.gpx").write_text("<gpx></gpx>")
            index[str(i)] = float(i)

        with index_path.open("w") as f:
            json.dump(index, f)

        ridewithgps._enforce_lru_limit()

        assert len(list(routes_dir.glob("*.gpx"))) == 5

    def test_enforce_lru_limit_over_limit(self, temp_cache_with_files):
        routes_dir = temp_cache_with_files["routes_dir"]
        index_path = temp_cache_with_files["index_path"]

        index = {}
        for i in range(12):
            (routes_dir / f"{i}.gpx").write_text("<gpx></gpx>")
            index[str(i)] = float(i)

        with index_path.open("w") as f:
            json.dump(index, f)

        ridewithgps._enforce_lru_limit()

        remaining_files = list(routes_dir.glob("*.gpx"))
        assert len(remaining_files) == 10

        with index_path.open() as f:
            new_index = json.load(f)
        assert len(new_index) == 10

        assert "0" not in new_index
        assert "1" not in new_index
        assert "11" in new_index

    def test_evicts_oldest_files(self, temp_cache_with_files):
        routes_dir = temp_cache_with_files["routes_dir"]
        index_path = temp_cache_with_files["index_path"]

        index = {
            "oldest": 100.0,
            "second_oldest": 200.0,
            "newest": 300.0,
        }
        for route_id in index:
            (routes_dir / f"{route_id}.gpx").write_text("<gpx></gpx>")

        with index_path.open("w") as f:
            json.dump(index, f)

        with patch.object(ridewithgps, "MAX_CACHED_ROUTES", 2):
            ridewithgps._enforce_lru_limit()

        assert not (routes_dir / "oldest.gpx").exists()
        assert (routes_dir / "second_oldest.gpx").exists()
        assert (routes_dir / "newest.gpx").exists()


class TestGetGpx:
    @pytest.fixture
    def temp_cache_dir(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / ".cache" / "gpx-analyzer"
        routes_dir = cache_dir / "routes"
        routes_dir.mkdir(parents=True)
        index_path = cache_dir / "cache_index.json"

        monkeypatch.setattr(ridewithgps, "CACHE_DIR", cache_dir)
        monkeypatch.setattr(ridewithgps, "ROUTES_DIR", routes_dir)
        monkeypatch.setattr(ridewithgps, "CACHE_INDEX_PATH", index_path)

        return {
            "cache_dir": cache_dir,
            "routes_dir": routes_dir,
            "index_path": index_path,
        }

    def test_cache_hit_returns_path(self, temp_cache_dir):
        routes_dir = temp_cache_dir["routes_dir"]
        index_path = temp_cache_dir["index_path"]

        gpx_file = routes_dir / "12345.gpx"
        gpx_file.write_text("<gpx>cached</gpx>")
        with index_path.open("w") as f:
            json.dump({"12345": 100.0}, f)

        result = ridewithgps.get_gpx("https://ridewithgps.com/routes/12345")

        assert result == str(gpx_file)

    def test_cache_hit_updates_lru(self, temp_cache_dir):
        routes_dir = temp_cache_dir["routes_dir"]
        index_path = temp_cache_dir["index_path"]

        gpx_file = routes_dir / "12345.gpx"
        gpx_file.write_text("<gpx>cached</gpx>")
        with index_path.open("w") as f:
            json.dump({"12345": 100.0}, f)

        ridewithgps.get_gpx("https://ridewithgps.com/routes/12345")

        with index_path.open() as f:
            index = json.load(f)
        assert index["12345"] > 100.0

    @patch.object(ridewithgps, "_download_gpx")
    def test_cache_miss_downloads(self, mock_download, temp_cache_dir):
        routes_dir = temp_cache_dir["routes_dir"]

        mock_download.return_value = b"<gpx>downloaded</gpx>"

        result = ridewithgps.get_gpx("https://ridewithgps.com/routes/99999")

        mock_download.assert_called_once_with(99999, None)
        assert result == str(routes_dir / "99999.gpx")
        assert (routes_dir / "99999.gpx").read_bytes() == b"<gpx>downloaded</gpx>"

    @patch.object(ridewithgps, "_download_gpx")
    def test_cache_miss_downloads_with_privacy_code(self, mock_download, temp_cache_dir):
        routes_dir = temp_cache_dir["routes_dir"]

        mock_download.return_value = b"<gpx>private route</gpx>"

        result = ridewithgps.get_gpx(
            "https://ridewithgps.com/routes/99999?privacy_code=SECRET"
        )

        mock_download.assert_called_once_with(99999, "SECRET")
        assert result == str(routes_dir / "99999.gpx")
        assert (routes_dir / "99999.gpx").read_bytes() == b"<gpx>private route</gpx>"


class TestLoadConfig:
    @pytest.fixture
    def no_local_config(self, tmp_path, monkeypatch):
        """Ensure no local config file exists."""
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

    def test_returns_config_from_global_file(self, tmp_path, monkeypatch, no_local_config):
        config_dir = tmp_path / ".config" / "gpx-analyzer"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "gpx-analyzer.json"
        config_path.write_text('{"ridewithgps_api_key": "global-key"}')

        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", config_path)

        config = ridewithgps._load_config()

        assert config == {"ridewithgps_api_key": "global-key"}

    def test_returns_config_from_local_file(self, tmp_path, monkeypatch):
        local_path = tmp_path / "gpx-analyzer.json"
        local_path.write_text('{"ridewithgps_api_key": "local-key"}')
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        # Global config also exists
        config_dir = tmp_path / ".config" / "gpx-analyzer"
        config_dir.mkdir(parents=True)
        global_path = config_dir / "gpx-analyzer.json"
        global_path.write_text('{"ridewithgps_api_key": "global-key"}')
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)

        config = ridewithgps._load_config()

        # Local takes precedence
        assert config == {"ridewithgps_api_key": "local-key"}

    def test_falls_back_to_global_when_local_missing(self, tmp_path, monkeypatch):
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        config_dir = tmp_path / ".config" / "gpx-analyzer"
        config_dir.mkdir(parents=True)
        global_path = config_dir / "gpx-analyzer.json"
        global_path.write_text('{"ridewithgps_api_key": "global-key"}')
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)

        config = ridewithgps._load_config()

        assert config == {"ridewithgps_api_key": "global-key"}

    def test_returns_empty_when_both_files_missing(self, tmp_path, monkeypatch):
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        global_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)

        config = ridewithgps._load_config()

        assert config == {}

    def test_skips_invalid_local_and_uses_global(self, tmp_path, monkeypatch):
        local_path = tmp_path / "gpx-analyzer.json"
        local_path.write_text("not valid json")
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        config_dir = tmp_path / ".config" / "gpx-analyzer"
        config_dir.mkdir(parents=True)
        global_path = config_dir / "gpx-analyzer.json"
        global_path.write_text('{"ridewithgps_api_key": "global-key"}')
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)

        config = ridewithgps._load_config()

        assert config == {"ridewithgps_api_key": "global-key"}

    def test_returns_empty_when_both_invalid(self, tmp_path, monkeypatch):
        local_path = tmp_path / "gpx-analyzer.json"
        local_path.write_text("not valid json")
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        config_dir = tmp_path / ".config" / "gpx-analyzer"
        config_dir.mkdir(parents=True)
        global_path = config_dir / "gpx-analyzer.json"
        global_path.write_text("also not valid")
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)

        config = ridewithgps._load_config()

        assert config == {}


class TestGetAuthHeaders:
    @pytest.fixture
    def no_config(self, tmp_path, monkeypatch):
        """Ensure no config files exist."""
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)
        global_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)

    def test_returns_headers_from_config_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)
        monkeypatch.delenv("RIDEWITHGPS_AUTH_TOKEN", raising=False)

        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        config_dir = tmp_path / ".config" / "gpx-analyzer"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "gpx-analyzer.json"
        config_path.write_text(
            '{"ridewithgps_api_key": "file-key", "ridewithgps_auth_token": "file-token"}'
        )
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", config_path)

        headers = ridewithgps._get_auth_headers()

        assert headers == {
            "x-rwgps-api-key": "file-key",
            "x-rwgps-auth-token": "file-token",
        }

    def test_returns_headers_from_env_vars(self, tmp_path, monkeypatch, no_config):
        monkeypatch.setenv("RIDEWITHGPS_API_KEY", "env-key")
        monkeypatch.setenv("RIDEWITHGPS_AUTH_TOKEN", "env-token")

        headers = ridewithgps._get_auth_headers()

        assert headers == {
            "x-rwgps-api-key": "env-key",
            "x-rwgps-auth-token": "env-token",
        }

    def test_config_file_takes_precedence_over_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RIDEWITHGPS_API_KEY", "env-key")
        monkeypatch.setenv("RIDEWITHGPS_AUTH_TOKEN", "env-token")

        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        config_dir = tmp_path / ".config" / "gpx-analyzer"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "gpx-analyzer.json"
        config_path.write_text(
            '{"ridewithgps_api_key": "file-key", "ridewithgps_auth_token": "file-token"}'
        )
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", config_path)

        headers = ridewithgps._get_auth_headers()

        assert headers == {
            "x-rwgps-api-key": "file-key",
            "x-rwgps-auth-token": "file-token",
        }

    def test_falls_back_to_env_for_missing_config_values(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RIDEWITHGPS_AUTH_TOKEN", "env-token")
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)

        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        config_dir = tmp_path / ".config" / "gpx-analyzer"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "gpx-analyzer.json"
        # Config only has api_key, not auth_token
        config_path.write_text('{"ridewithgps_api_key": "file-key"}')
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", config_path)

        headers = ridewithgps._get_auth_headers()

        assert headers == {
            "x-rwgps-api-key": "file-key",
            "x-rwgps-auth-token": "env-token",
        }

    def test_returns_empty_when_api_key_missing(self, tmp_path, monkeypatch, no_config):
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)
        monkeypatch.setenv("RIDEWITHGPS_AUTH_TOKEN", "my-auth-token")

        headers = ridewithgps._get_auth_headers()

        assert headers == {}

    def test_returns_empty_when_auth_token_missing(self, tmp_path, monkeypatch, no_config):
        monkeypatch.setenv("RIDEWITHGPS_API_KEY", "my-api-key")
        monkeypatch.delenv("RIDEWITHGPS_AUTH_TOKEN", raising=False)

        headers = ridewithgps._get_auth_headers()

        assert headers == {}

    def test_returns_empty_when_both_missing(self, tmp_path, monkeypatch, no_config):
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)
        monkeypatch.delenv("RIDEWITHGPS_AUTH_TOKEN", raising=False)

        headers = ridewithgps._get_auth_headers()

        assert headers == {}


class TestDownloadGpx:
    @pytest.fixture
    def no_config(self, tmp_path, monkeypatch):
        """Ensure no config files exist and no env vars are set."""
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)
        global_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)
        monkeypatch.delenv("RIDEWITHGPS_AUTH_TOKEN", raising=False)

    @patch("gpx_analyzer.ridewithgps.requests.get")
    def test_download_success(self, mock_get, no_config):
        mock_response = MagicMock()
        mock_response.content = b"<gpx>downloaded content</gpx>"
        mock_get.return_value = mock_response

        result = ridewithgps._download_gpx(12345)

        mock_get.assert_called_once_with(
            "https://ridewithgps.com/routes/12345.gpx?sub_format=track",
            headers={},
            timeout=30,
        )
        mock_response.raise_for_status.assert_called_once()
        assert result == b"<gpx>downloaded content</gpx>"

    @patch("gpx_analyzer.ridewithgps.requests.get")
    def test_download_with_privacy_code(self, mock_get, no_config):
        mock_response = MagicMock()
        mock_response.content = b"<gpx>private content</gpx>"
        mock_get.return_value = mock_response

        result = ridewithgps._download_gpx(12345, "SECRET123")

        mock_get.assert_called_once_with(
            "https://ridewithgps.com/routes/12345.gpx?sub_format=track&privacy_code=SECRET123",
            headers={},
            timeout=30,
        )
        mock_response.raise_for_status.assert_called_once()
        assert result == b"<gpx>private content</gpx>"

    @patch("gpx_analyzer.ridewithgps.requests.get")
    def test_download_with_auth_from_env(self, mock_get, tmp_path, monkeypatch):
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)
        global_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)
        monkeypatch.setenv("RIDEWITHGPS_API_KEY", "test-key")
        monkeypatch.setenv("RIDEWITHGPS_AUTH_TOKEN", "test-token")

        mock_response = MagicMock()
        mock_response.content = b"<gpx>authenticated content</gpx>"
        mock_get.return_value = mock_response

        result = ridewithgps._download_gpx(12345)

        mock_get.assert_called_once_with(
            "https://ridewithgps.com/routes/12345.gpx?sub_format=track",
            headers={
                "x-rwgps-api-key": "test-key",
                "x-rwgps-auth-token": "test-token",
            },
            timeout=30,
        )
        assert result == b"<gpx>authenticated content</gpx>"

    @patch("gpx_analyzer.ridewithgps.requests.get")
    def test_download_with_auth_from_global_config(self, mock_get, tmp_path, monkeypatch):
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        config_dir = tmp_path / ".config" / "gpx-analyzer"
        config_dir.mkdir(parents=True)
        global_path = config_dir / "gpx-analyzer.json"
        global_path.write_text(
            '{"ridewithgps_api_key": "cfg-key", "ridewithgps_auth_token": "cfg-token"}'
        )
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)
        monkeypatch.delenv("RIDEWITHGPS_AUTH_TOKEN", raising=False)

        mock_response = MagicMock()
        mock_response.content = b"<gpx>authenticated content</gpx>"
        mock_get.return_value = mock_response

        result = ridewithgps._download_gpx(12345)

        mock_get.assert_called_once_with(
            "https://ridewithgps.com/routes/12345.gpx?sub_format=track",
            headers={
                "x-rwgps-api-key": "cfg-key",
                "x-rwgps-auth-token": "cfg-token",
            },
            timeout=30,
        )
        assert result == b"<gpx>authenticated content</gpx>"

    @patch("gpx_analyzer.ridewithgps.requests.get")
    def test_download_with_auth_from_local_config(self, mock_get, tmp_path, monkeypatch):
        local_path = tmp_path / "gpx-analyzer.json"
        local_path.write_text(
            '{"ridewithgps_api_key": "local-key", "ridewithgps_auth_token": "local-token"}'
        )
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)

        global_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)
        monkeypatch.delenv("RIDEWITHGPS_AUTH_TOKEN", raising=False)

        mock_response = MagicMock()
        mock_response.content = b"<gpx>authenticated content</gpx>"
        mock_get.return_value = mock_response

        result = ridewithgps._download_gpx(12345)

        mock_get.assert_called_once_with(
            "https://ridewithgps.com/routes/12345.gpx?sub_format=track",
            headers={
                "x-rwgps-api-key": "local-key",
                "x-rwgps-auth-token": "local-token",
            },
            timeout=30,
        )
        assert result == b"<gpx>authenticated content</gpx>"

    @patch("gpx_analyzer.ridewithgps.requests.get")
    def test_download_failure(self, mock_get, no_config):
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            ridewithgps._download_gpx(99999)


class TestSurfaceCrrDeltas:
    """Tests for surface crr delta calculations."""

    @pytest.fixture(autouse=True)
    def use_default_config(self, tmp_path, monkeypatch):
        """Ensure tests use default code values, not config file values."""
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", tmp_path / "nonexistent1.json")
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", tmp_path / "nonexistent2.json")

    def test_paved_quality_baseline(self):
        """R=3 (quality paved) with S=0 (paved) should return baseline crr."""
        baseline = 0.004
        assert ridewithgps._surface_type_to_crr(3, 0, baseline) == 0.004

    def test_paved_standard(self):
        """R=4 (standard paved) with S=0 should add delta of 0.001."""
        baseline = 0.004
        assert ridewithgps._surface_type_to_crr(4, 0, baseline) == 0.005

    def test_gravel(self):
        """R=15 (gravel) with S=0 should add delta of 0.006."""
        baseline = 0.004
        assert ridewithgps._surface_type_to_crr(15, 0, baseline) == 0.010

    def test_rough_gravel(self):
        """R=25 (rough gravel) with S=0 should add delta of 0.008."""
        baseline = 0.004
        assert ridewithgps._surface_type_to_crr(25, 0, baseline) == 0.012

    def test_unknown_r_returns_baseline(self):
        """Unknown R values with S=0 should return baseline (delta=0)."""
        baseline = 0.005
        assert ridewithgps._surface_type_to_crr(999, 0, baseline) == 0.005

    def test_none_r_returns_baseline(self):
        """None R value with S=0 should return baseline."""
        baseline = 0.006
        assert ridewithgps._surface_type_to_crr(None, 0, baseline) == 0.006

    def test_different_baseline(self):
        """Deltas should apply correctly with different baseline values."""
        baseline = 0.010
        assert ridewithgps._surface_type_to_crr(3, 0, baseline) == 0.010   # delta=0
        assert ridewithgps._surface_type_to_crr(4, 0, baseline) == 0.011   # delta=0.001
        assert ridewithgps._surface_type_to_crr(15, 0, baseline) == 0.016  # delta=0.006

    def test_unpaved_s_adds_delta(self):
        """S >= 50 should add unpaved penalty."""
        baseline = 0.004
        # R=4 with S=0: 0.004 + 0.001 = 0.005
        assert ridewithgps._surface_type_to_crr(4, 0, baseline) == 0.005
        # R=4 with S=56 (unpaved): 0.004 + 0.001 + 0.005 = 0.010
        assert ridewithgps._surface_type_to_crr(4, 56, baseline) == 0.010

    def test_unpaved_s_threshold(self):
        """S values below 50 should not add unpaved penalty."""
        baseline = 0.004
        assert ridewithgps._surface_type_to_crr(4, 49, baseline) == 0.005  # no penalty
        assert ridewithgps._surface_type_to_crr(4, 50, baseline) == 0.010  # penalty added


class TestSurfaceCrrDeltasConfig:
    """Tests for config file override of surface crr deltas."""

    def test_config_overrides_deltas(self, tmp_path, monkeypatch):
        """Config file can override surface_crr_deltas."""
        config_path = tmp_path / "gpx-analyzer.json"
        config_path.write_text('{"surface_crr_deltas": {"3": 0.002, "15": 0.010}}')
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", config_path)
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", tmp_path / "nonexistent.json")

        baseline = 0.004
        # With custom deltas: R=3 has delta 0.002, R=15 has delta 0.010
        assert ridewithgps._surface_type_to_crr(3, 0, baseline) == 0.006   # 0.004 + 0.002
        assert ridewithgps._surface_type_to_crr(15, 0, baseline) == 0.014  # 0.004 + 0.010
        # Unknown R value still uses delta=0
        assert ridewithgps._surface_type_to_crr(999, 0, baseline) == 0.004

    def test_no_config_uses_defaults(self, tmp_path, monkeypatch):
        """Without config, default deltas are used."""
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", tmp_path / "nonexistent1.json")
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", tmp_path / "nonexistent2.json")

        baseline = 0.004
        assert ridewithgps._surface_type_to_crr(3, 0, baseline) == 0.004   # delta=0
        assert ridewithgps._surface_type_to_crr(15, 0, baseline) == 0.010  # delta=0.006

    def test_config_overrides_unpaved_delta(self, tmp_path, monkeypatch):
        """Config file can override unpaved_crr_delta."""
        config_path = tmp_path / "gpx-analyzer.json"
        config_path.write_text('{"unpaved_crr_delta": 0.010}')
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", config_path)
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", tmp_path / "nonexistent.json")

        baseline = 0.004
        # R=4 with S=56 (unpaved): 0.004 + 0.001 + 0.010 = 0.015
        assert ridewithgps._surface_type_to_crr(4, 56, baseline) == 0.015


class TestIsUnpaved:
    """Tests for is_unpaved() function."""

    def test_s_below_threshold_is_paved(self):
        assert ridewithgps.is_unpaved(0) is False
        assert ridewithgps.is_unpaved(1) is False
        assert ridewithgps.is_unpaved(49) is False

    def test_s_at_threshold_is_unpaved(self):
        assert ridewithgps.is_unpaved(50) is True

    def test_s_above_threshold_is_unpaved(self):
        assert ridewithgps.is_unpaved(56) is True
        assert ridewithgps.is_unpaved(63) is True
        assert ridewithgps.is_unpaved(99) is True

    def test_none_is_paved(self):
        assert ridewithgps.is_unpaved(None) is False


class TestParseJsonTrackPoints:
    @pytest.fixture(autouse=True)
    def use_default_config(self, tmp_path, monkeypatch):
        """Ensure tests use default code values, not config file values."""
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", tmp_path / "nonexistent1.json")
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", tmp_path / "nonexistent2.json")

    def test_empty_route_data(self):
        result = ridewithgps.parse_json_track_points({}, baseline_crr=0.004)
        assert result == []

    def test_empty_track_points(self):
        result = ridewithgps.parse_json_track_points({"track_points": []}, baseline_crr=0.004)
        assert result == []

    def test_basic_track_points_no_surface(self):
        """Track points without R value should have crr=None."""
        route_data = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 10.0, "d": 0},
                {"x": -122.4183, "y": 37.7758, "e": 15.0, "d": 100},
            ],
        }
        result = ridewithgps.parse_json_track_points(route_data, baseline_crr=0.004)
        assert len(result) == 2
        assert result[0].lat == 37.7749
        assert result[0].lon == -122.4194
        assert result[0].elevation == 10.0
        assert result[0].crr is None  # No R value

    def test_track_points_with_r_values(self):
        """Track points with R values should have crr = baseline + delta."""
        route_data = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 10.0, "d": 0, "R": 4},
                {"x": -122.4183, "y": 37.7758, "e": 15.0, "d": 100, "R": 4},
                {"x": -122.4172, "y": 37.7767, "e": 20.0, "d": 200, "R": 15},
            ],
        }
        result = ridewithgps.parse_json_track_points(route_data, baseline_crr=0.004)
        assert len(result) == 3
        assert result[0].crr == 0.005  # R=4: 0.004 + 0.001
        assert result[1].crr == 0.005  # R=4: 0.004 + 0.001
        assert result[2].crr == 0.010  # R=15: 0.004 + 0.006

    def test_track_points_missing_lat_lon_skipped(self):
        route_data = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 10.0, "d": 0, "R": 4},
                {"x": None, "y": 37.7758, "e": 15.0, "d": 100, "R": 4},  # Missing lon
                {"x": -122.4172, "e": 20.0, "d": 200, "R": 15},           # Missing lat
            ],
        }
        result = ridewithgps.parse_json_track_points(route_data, baseline_crr=0.004)
        assert len(result) == 1
        assert result[0].lat == 37.7749

    def test_nested_route_format(self):
        """Handle nested route format (for backwards compatibility)."""
        route_data = {
            "route": {
                "track_points": [
                    {"x": -122.4194, "y": 37.7749, "e": 10.0, "R": 4},
                ],
            }
        }
        result = ridewithgps.parse_json_track_points(route_data, baseline_crr=0.004)
        assert len(result) == 1
        assert result[0].crr == 0.005  # 0.004 + 0.001

    def test_different_baseline_crr(self):
        """Verify deltas are applied to different baseline values."""
        route_data = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 10.0, "R": 3, "S": 0},
                {"x": -122.4183, "y": 37.7758, "e": 15.0, "R": 15, "S": 0},
            ],
        }
        result = ridewithgps.parse_json_track_points(route_data, baseline_crr=0.010)
        assert result[0].crr == 0.010  # R=3: 0.010 + 0 (baseline)
        assert result[1].crr == 0.016  # R=15: 0.010 + 0.006

    def test_unpaved_flag_set_from_s_value(self):
        """Verify unpaved flag is set based on S value >= 50."""
        route_data = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 10.0, "R": 4, "S": 0},   # paved
                {"x": -122.4183, "y": 37.7758, "e": 15.0, "R": 4, "S": 49},  # paved (below threshold)
                {"x": -122.4172, "y": 37.7767, "e": 20.0, "R": 5, "S": 56},  # unpaved
                {"x": -122.4161, "y": 37.7776, "e": 25.0, "R": 5, "S": 63},  # unpaved
            ],
        }
        result = ridewithgps.parse_json_track_points(route_data, baseline_crr=0.004)
        assert result[0].unpaved is False
        assert result[1].unpaved is False
        assert result[2].unpaved is True
        assert result[3].unpaved is True

    def test_unpaved_adds_crr_penalty(self):
        """Verify unpaved surfaces get additional crr penalty."""
        route_data = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 10.0, "R": 4, "S": 0},   # paved
                {"x": -122.4183, "y": 37.7758, "e": 15.0, "R": 4, "S": 56},  # unpaved
            ],
        }
        result = ridewithgps.parse_json_track_points(route_data, baseline_crr=0.004)
        # Paved: baseline + R_delta = 0.004 + 0.001 = 0.005
        assert result[0].crr == 0.005
        # Unpaved: baseline + R_delta + unpaved_delta = 0.004 + 0.001 + 0.005 = 0.010
        assert result[1].crr == 0.010


class TestDownloadJson:
    @pytest.fixture
    def no_config(self, tmp_path, monkeypatch):
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)
        global_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)
        monkeypatch.delenv("RIDEWITHGPS_AUTH_TOKEN", raising=False)

    @patch("gpx_analyzer.ridewithgps.requests.get")
    def test_download_json_success(self, mock_get, no_config):
        mock_response = MagicMock()
        mock_response.json.return_value = {"route": {"name": "Test Route"}}
        mock_get.return_value = mock_response

        result = ridewithgps._download_json(12345)

        mock_get.assert_called_once_with(
            "https://ridewithgps.com/routes/12345.json",
            headers={},
            timeout=30,
        )
        assert result == {"route": {"name": "Test Route"}}

    @patch("gpx_analyzer.ridewithgps.requests.get")
    def test_download_json_with_privacy_code(self, mock_get, no_config):
        mock_response = MagicMock()
        mock_response.json.return_value = {"route": {"name": "Private Route"}}
        mock_get.return_value = mock_response

        result = ridewithgps._download_json(12345, "SECRET123")

        mock_get.assert_called_once_with(
            "https://ridewithgps.com/routes/12345.json?privacy_code=SECRET123",
            headers={},
            timeout=30,
        )
        assert result == {"route": {"name": "Private Route"}}


class TestGetRouteWithSurface:
    @pytest.fixture
    def no_config(self, tmp_path, monkeypatch):
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)
        global_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)
        monkeypatch.delenv("RIDEWITHGPS_AUTH_TOKEN", raising=False)

    @patch.object(ridewithgps, "_save_route_json_to_cache")
    @patch.object(ridewithgps, "_load_cached_route_json", return_value=None)
    @patch.object(ridewithgps, "_download_json")
    def test_returns_points_and_metadata(self, mock_download, mock_cache_load, mock_cache_save, no_config):
        mock_download.return_value = {
            "name": "Test Route",
            "distance": 10000,
            "elevation_gain": 100,
            "unpaved_pct": 15,
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 10.0, "d": 0, "R": 4},
                {"x": -122.4183, "y": 37.7758, "e": 15.0, "d": 100, "R": 4},
            ],
            "surface": "mostly_paved",
        }

        points, metadata = ridewithgps.get_route_with_surface(
            "https://ridewithgps.com/routes/12345", baseline_crr=0.004
        )

        assert len(points) == 2
        assert points[0].lat == 37.7749
        assert points[0].crr == 0.005  # 0.004 + 0.001
        assert metadata["name"] == "Test Route"
        assert metadata["unpaved_pct"] == 15

    @patch.object(ridewithgps, "_save_route_json_to_cache")
    @patch.object(ridewithgps, "_load_cached_route_json", return_value=None)
    @patch.object(ridewithgps, "_download_json")
    def test_extracts_privacy_code(self, mock_download, mock_cache_load, mock_cache_save, no_config):
        mock_download.return_value = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 10.0, "d": 0, "R": 4},
            ],
        }

        ridewithgps.get_route_with_surface(
            "https://ridewithgps.com/routes/12345?privacy_code=SECRET", baseline_crr=0.004
        )

        mock_download.assert_called_once_with(12345, "SECRET")

    @patch.object(ridewithgps, "_save_route_json_to_cache")
    @patch.object(ridewithgps, "_load_cached_route_json", return_value=None)
    @patch.object(ridewithgps, "_download_json")
    def test_uses_baseline_crr(self, mock_download, mock_cache_load, mock_cache_save, no_config):
        """Verify baseline_crr is used correctly in crr calculation."""
        mock_download.return_value = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 10.0, "R": 3},   # baseline
                {"x": -122.4183, "y": 37.7758, "e": 15.0, "R": 15},  # gravel
            ],
        }

        points, _ = ridewithgps.get_route_with_surface(
            "https://ridewithgps.com/routes/12345", baseline_crr=0.008
        )

        assert points[0].crr == 0.008  # R=3: baseline + 0
        assert points[1].crr == 0.014  # R=15: baseline + 0.006


class TestIsRidewithgpsTripUrl:
    def test_valid_trip_url(self):
        assert ridewithgps.is_ridewithgps_trip_url("https://ridewithgps.com/trips/233763291")

    def test_valid_trip_url_with_www(self):
        assert ridewithgps.is_ridewithgps_trip_url(
            "https://www.ridewithgps.com/trips/233763291"
        )

    def test_valid_trip_url_http(self):
        assert ridewithgps.is_ridewithgps_trip_url("http://ridewithgps.com/trips/233763291")

    def test_route_url_not_trip(self):
        assert not ridewithgps.is_ridewithgps_trip_url("https://ridewithgps.com/routes/12345")

    def test_local_file_path(self):
        assert not ridewithgps.is_ridewithgps_trip_url("/path/to/file.gpx")

    def test_empty_string(self):
        assert not ridewithgps.is_ridewithgps_trip_url("")


class TestExtractTripId:
    def test_valid_url(self):
        assert (
            ridewithgps.extract_trip_id("https://ridewithgps.com/trips/233763291")
            == 233763291
        )

    def test_valid_url_with_www(self):
        assert (
            ridewithgps.extract_trip_id("https://www.ridewithgps.com/trips/12345")
            == 12345
        )

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid RideWithGPS trip URL"):
            ridewithgps.extract_trip_id("https://ridewithgps.com/routes/123")

    def test_local_path_raises(self):
        with pytest.raises(ValueError, match="Invalid RideWithGPS trip URL"):
            ridewithgps.extract_trip_id("/path/to/file.gpx")


class TestParseTripTrackPoints:
    def test_parses_basic_fields(self):
        trip_data = {
            "trip": {
                "track_points": [
                    {"x": -122.4194, "y": 37.7749, "e": 100.0, "d": 0.0, "s": 5.0, "t": 1000000},
                    {"x": -122.4183, "y": 37.7758, "e": 105.0, "d": 100.0, "s": 6.0, "t": 1000020},
                ]
            }
        }
        points = ridewithgps.parse_trip_track_points(trip_data)

        assert len(points) == 2
        assert points[0].lat == 37.7749
        assert points[0].lon == -122.4194
        assert points[0].elevation == 100.0
        assert points[0].distance == 0.0
        assert points[0].speed == 5.0
        assert points[0].timestamp == 1000000

    def test_parses_power_hr_cadence(self):
        trip_data = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749, "e": 100.0, "d": 0.0, "s": 5.0, "p": 150, "h": 130, "c": 80},
            ]
        }
        points = ridewithgps.parse_trip_track_points(trip_data)

        assert points[0].power == 150
        assert points[0].heart_rate == 130
        assert points[0].cadence == 80

    def test_handles_missing_optional_fields(self):
        trip_data = {
            "track_points": [
                {"x": -122.4194, "y": 37.7749},  # Only lat/lon
            ]
        }
        points = ridewithgps.parse_trip_track_points(trip_data)

        assert len(points) == 1
        assert points[0].elevation is None
        assert points[0].speed is None
        assert points[0].power is None

    def test_skips_points_without_lat_lon(self):
        trip_data = {
            "track_points": [
                {"x": -122.4194},  # Missing lat
                {"y": 37.7749},  # Missing lon
                {"x": -122.4194, "y": 37.7749},  # Valid
            ]
        }
        points = ridewithgps.parse_trip_track_points(trip_data)

        assert len(points) == 1

    def test_empty_track_points(self):
        trip_data = {"track_points": []}
        points = ridewithgps.parse_trip_track_points(trip_data)
        assert points == []

    def test_missing_track_points(self):
        trip_data = {"name": "Test Trip"}
        points = ridewithgps.parse_trip_track_points(trip_data)
        assert points == []


class TestGetTripData:
    @pytest.fixture
    def no_config(self, tmp_path, monkeypatch):
        local_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "LOCAL_CONFIG_PATH", local_path)
        global_path = tmp_path / "nonexistent" / "gpx-analyzer.json"
        monkeypatch.setattr(ridewithgps, "CONFIG_PATH", global_path)
        monkeypatch.delenv("RIDEWITHGPS_API_KEY", raising=False)
        monkeypatch.delenv("RIDEWITHGPS_AUTH_TOKEN", raising=False)

    @patch.object(ridewithgps, "_save_trip_to_cache")
    @patch.object(ridewithgps, "_load_cached_trip", return_value=None)
    @patch.object(ridewithgps, "_download_trip_json")
    def test_returns_points_and_metadata(self, mock_download, mock_cache_load, mock_cache_save, no_config):
        mock_download.return_value = {
            "trip": {
                "name": "Test Ride",
                "distance": 50000,
                "elevation_gain": 500,
                "moving_time": 7200,
                "duration": 8000,
                "avg_speed": 6.9,
                "avg_watts": 120,
                "track_points": [
                    {"x": -122.4194, "y": 37.7749, "e": 100.0, "d": 0.0, "s": 5.0},
                    {"x": -122.4183, "y": 37.7758, "e": 105.0, "d": 100.0, "s": 6.0},
                ],
            }
        }

        points, metadata = ridewithgps.get_trip_data(
            "https://ridewithgps.com/trips/233763291"
        )

        assert len(points) == 2
        assert points[0].lat == 37.7749
        assert metadata["name"] == "Test Ride"
        assert metadata["distance"] == 50000
        assert metadata["moving_time"] == 7200
        assert metadata["avg_watts"] == 120

    @patch.object(ridewithgps, "_download_trip_json")
    def test_invalid_url_raises(self, mock_download, no_config):
        with pytest.raises(ValueError, match="Invalid RideWithGPS trip URL"):
            ridewithgps.get_trip_data("https://ridewithgps.com/routes/12345")
