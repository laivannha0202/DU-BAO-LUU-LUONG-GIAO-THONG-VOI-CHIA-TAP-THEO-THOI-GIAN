/* ==========================================================================
   app/static/js/predict.js
   Màn 2 — Dự báo. Gọi thật POST /api/traffic-forecast.
   Không hard-code bất kỳ giá trị dự báo hay metric nào.
   ========================================================================== */
(function () {
  "use strict";

  var API_FORECAST = "/api/traffic-forecast";
  var API_MODEL_INFO = "/api/model-info";

  var $ = function (id) { return document.getElementById(id); };

  var form = $("forecast-form");
  var statusArea = $("status-area");
  var resultArea = $("result-area");
  var submitBtn = $("submit-btn");
  var resetBtn = $("reset-btn");
  var weatherList = $("weather-list");
  var modelHint = $("model-hint");

  var DEFAULTS = { date: "2018-06-15", time: "08:00", temperature: "20.0", clouds: "20", rain: "0.0", snow: "0.0" };
  var FIELD_IDS = ["date", "time", "temperature", "clouds", "rain", "snow", "weather", "state-fair"];

  /* ---------------- tiện ích ---------------- */
  function clearFieldErrors() {
    FIELD_IDS.forEach(function (id) {
      var input = $(id);
      var err = $(id + "-err");
      if (err) { err.hidden = true; err.textContent = ""; }
      if (input && input.setAttribute) { input.removeAttribute("aria-invalid"); }
    });
  }

  function setFieldError(id, message) {
    var err = $(id + "-err");
    var input = $(id);
    if (err) { err.textContent = message; err.hidden = false; }
    if (input && input.setAttribute) { input.setAttribute("aria-invalid", "true"); }
  }

  function clearAreas() { statusArea.innerHTML = ""; resultArea.innerHTML = ""; }

  function esc(text) {
    return String(text == null ? "" : text)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function fmt(n, digits) {
    if (n === null || n === undefined || isNaN(n)) return "—";
    return Number(n).toLocaleString("vi-VN", {
      minimumFractionDigits: digits, maximumFractionDigits: digits
    });
  }

  /* ---------------- danh mục thời tiết (lấy từ API, không hard-code) ---------------- */
  function loadWeatherCategories() {
    fetch(API_MODEL_INFO, { headers: { "Accept": "application/json" } })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (info) {
        var allowed = (info.serving && info.serving.allowed_weather) || [];
        weatherList.innerHTML = "";
        if (!allowed.length) {
          weatherList.innerHTML = '<p class="small err">Không lấy được danh mục thời tiết từ server.</p>';
          return;
        }
        allowed.forEach(function (name) {
          var id = "w-" + name.toLowerCase();
          var label = document.createElement("label");
          label.className = "check";
          label.htmlFor = id;
          var input = document.createElement("input");
          input.type = "checkbox";
          input.id = id;
          input.name = "weather";
          input.value = name;
          var span = document.createElement("span");
          span.textContent = name;
          label.appendChild(input);
          label.appendChild(span);
          weatherList.appendChild(label);
        });
        var hp = info.model && info.model.hyperparameters;
        if (hp && hp.alpha !== undefined && hp.alpha !== null) {
          modelHint.textContent =
            "Mô hình: Ridge alpha=" + hp.alpha + " · target: " +
            ((info.model && info.model.target) || "traffic_volume");
        }
      })
      .catch(function (err) {
        weatherList.innerHTML =
          '<p class="small err">Không tải được danh mục thời tiết (' + esc(err.message) +
          "). Hãy kiểm tra server có đang chạy không.</p>";
      });
  }

  /* ---------------- kiểm tra phía trình duyệt ----------------
     Đây chỉ là kiểm tra tiện ích để phản hồi nhanh; nguồn chuẩn vẫn là Pydantic
     ở server (mọi lỗi vẫn được server trả lại đúng chuẩn HTTP 4xx).            */
  function validateClient() {
    var ok = true;
    var dateVal = $("date").value;
    var timeVal = $("time").value;
    var temp = parseFloat($("temperature").value);
    var cloudsRaw = $("clouds").value;
    var clouds = parseInt(cloudsRaw, 10);
    var rain = $("rain").value === "" ? 0 : parseFloat($("rain").value);
    var snow = $("snow").value === "" ? 0 : parseFloat($("snow").value);
    var checked = weatherList.querySelectorAll('input[name="weather"]:checked');

    if (!/^\d{4}-\d{2}-\d{2}$/.test(dateVal)) {
      setFieldError("date", "Vui lòng nhập ngày hợp lệ."); ok = false;
    }
    if (!/^\d{2}:\d{2}$/.test(timeVal)) {
      setFieldError("time", "Vui lòng nhập giờ hợp lệ (0–23)."); ok = false;
    }
    if (isNaN(temp)) {
      setFieldError("temperature", "Nhiệt độ phải là một số."); ok = false;
    } else if (temp < -70 || temp > 70) {
      setFieldError("temperature",
        "Nhiệt độ phải nằm trong khoảng −70 °C … 70 °C. Nhập ngoài khoảng này gần như chắc chắn là nhập nhầm đơn vị.");
      ok = false;
    }
    if (cloudsRaw === "" || isNaN(clouds) || !/^\d+$/.test(cloudsRaw.trim())) {
      setFieldError("clouds", "Độ phủ mây phải là số nguyên."); ok = false;
    } else if (clouds < 0 || clouds > 100) {
      setFieldError("clouds", "Độ phủ mây phải nằm trong khoảng 0 … 100."); ok = false;
    }
    if (isNaN(rain) || rain < 0) { setFieldError("rain", "Lượng mưa phải ≥ 0 mm."); ok = false; }
    if (isNaN(snow) || snow < 0) { setFieldError("snow", "Lượng tuyết phải ≥ 0 mm."); ok = false; }
    if (!checked.length) {
      setFieldError("weather", "Chọn ít nhất một hiện tượng thời tiết."); ok = false;
    }

    var fair = $("state-fair").value;
    if (fair && dateVal && fair.slice(0, 4) !== dateVal.slice(0, 4)) {
      setFieldError("state-fair", "Ngày State Fair phải cùng năm với ngày dự báo."); ok = false;
    }
    return ok;
  }

  function buildBody() {
    var checked = weatherList.querySelectorAll('input[name="weather"]:checked');
    var weather = Array.prototype.map.call(checked, function (c) { return c.value; }).sort();
    var body = {
      date_time: $("date").value + "T" + $("time").value + ":00",
      temperature_celsius: parseFloat($("temperature").value),
      clouds_all: parseInt($("clouds").value, 10),
      rain_1h_mm: $("rain").value === "" ? 0 : parseFloat($("rain").value),
      snow_1h_mm: $("snow").value === "" ? 0 : parseFloat($("snow").value),
      weather: weather
    };
    if ($("state-fair").value) { body.state_fair_start_date = $("state-fair").value; }
    return body;
  }

  /* ---------------- hiển thị ---------------- */
  function renderWarnings(warnings) {
    if (!warnings || !warnings.length) return "";
    return '<div class="alert alert-warn"><p><strong>Cảnh báo về phạm vi:</strong></p><ul>' +
      warnings.map(function (w) { return "<li>" + esc(w) + "</li>"; }).join("") +
      "</ul></div>";
  }

  function renderResult(data) {
    var features = data.derived_features || {};
    var holiday = data.holiday || {};
    var weather = data.weather || {};
    var cal = data.calendar || {};
    var multiHot = weather.multi_hot || {};
    var on = Object.keys(multiHot).filter(function (k) { return Number(multiHot[k]) === 1; });

    var rows = [
      ["Thời điểm dự báo", esc(data.input_echo.date_time)],
      ["Lịch (tự suy ra)",
        (cal.hour !== undefined ? String(cal.hour).padStart(2, "0") + ":00" : "—") +
        " · " + esc(cal.day_of_week_name || "?") +
        (cal.is_weekend ? " (cuối tuần)" : "") +
        " · tháng " + (cal.month !== undefined ? esc(cal.month) : "—")],
      ["Nhiệt độ",
        fmt(data.input_echo.temperature_celsius, 1) + " °C (= " +
        fmt(features.temp, 2) + " K sau khi chuyển đổi)"],
      ["Mưa / tuyết 1 giờ",
        fmt(data.input_echo.rain_1h_mm, 1) + " mm / " + fmt(data.input_echo.snow_1h_mm, 1) + " mm"],
      ["Độ phủ mây", fmt(data.input_echo.clouds_all, 0) + " %"],
      ["Thời tiết (multi-hot)",
        on.length ? on.map(function (k) { return esc(k.replace(/^wm_/, "")); }).join(", ") : "—"],
      ["Nhãn đại diện / gia đình / mức độ",
        esc(weather.weather_main_mode) + " · " + esc(weather.weather_family) +
        " · mức " + fmt(weather.weather_severity, 0)],
      ["Ngày lễ (tự tính từ lịch, không cần nhập tay)",
        (holiday.is_holiday ? "Có" : "Không") +
        (holiday.holiday_name && holiday.holiday_name !== "None"
          ? " — " + esc(holiday.holiday_name) : "")],
      ["<strong>RAW MODEL</strong> — Ridge",
        esc(data.model.name) + " (alpha=" + esc(data.model.alpha) +
        ", solver=" + esc(data.model.solver) + ") → " + fmt(data.raw_model_output, 2)],
      ["<strong>DEPLOYED PREDICTOR</strong> — max(0, ·) ∘ Ridge",
        (esc((data.predictor || {}).deployed_predictor || "max(0, Ridge)")) +
        (data.clipped_to_zero
          ? " → " + fmt(data.raw_model_output, 2) + " bị chiếu về <strong>0</strong>"
          : " → " + fmt(data.raw_model_output, 2) + " (không bị chiếu)")]
    ];

    return '' +
      renderWarnings(data.warnings) +
      '<div class="result" role="status">' +
      '  <p class="small muted mt0 mb0">DEPLOYED PREDICTOR — dự báo tại ' +
        esc(data.input_echo.date_time) + '</p>' +
      '  <div class="value">' + fmt(data.predicted_traffic_volume, 0) + '</div>' +
      '  <div class="unit">' + esc(data.unit_vi || "xe/giờ") + ' · ' + esc(data.unit) + '</div>' +
      '  <p class="small muted" style="margin-top:.4rem;margin-bottom:0">' +
        'RAW MODEL (Ridge) trả về <strong>' + fmt(data.raw_model_output, 2) + '</strong>' +
        (data.clipped_to_zero
          ? ' → được chiếu về sàn <strong>0</strong> (lưu lượng không thể âm)'
          : ' (giữ nguyên)') +
        '</p>' +
      '</div>' +
      '<div class="card">' +
      '  <h2 class="mt0">Chi tiết</h2>' +
      '  <div class="table-wrap"><table>' +
      '    <caption>Input đã dùng và những gì backend tự suy ra</caption><tbody>' +
      rows.map(function (r) {
        return '<tr><th scope="row">' + r[0] + '</th><td>' + r[1] + '</td></tr>';
      }).join("") +
      '  </tbody></table></div>' +
      '  <p class="small muted mb0">Đơn vị đầu ra: <code>traffic_volume</code> — lưu lượng, ' +
      '      đơn vị <strong>' + esc(data.unit_vi) + '</strong>. Giá trị luôn là số hữu hạn và không âm.' +
      '      <br><strong>RAW MODEL</strong> và <strong>DEPLOYED PREDICTOR</strong> là hai thứ khác nhau: ' +
      '      metric của <em>mô hình</em> trong báo cáo tính trên RAW; giá trị ở đây là của ' +
      '      <em>wrapper phục vụ</em>.</p>' +
      '</div>';
  }

  function renderError(payload, httpStatus) {
    var details = "";
    if (payload && payload.details && payload.details.length) {
      details = '<ul>' + payload.details.map(function (d) {
        return "<li><code>" + esc(d.field) + "</code>: " + esc(d.message) + "</li>";
      }).join("") + "</ul>";
    }
    statusArea.innerHTML =
      '<div class="alert alert-err" role="alert">' +
      '<p><strong>Không dự báo được (HTTP ' + esc(httpStatus) + ').</strong> ' +
      esc((payload && payload.message) || "Lỗi không xác định.") + "</p>" +
      details + "</div>";
    resultArea.innerHTML = "";
  }

  function renderLoading() {
    statusArea.innerHTML =
      '<div class="alert alert-info" role="status"><span class="spinner"></span>' +
      'Đang gửi yêu cầu tới <code>POST /api/traffic-forecast</code>…</div>';
    resultArea.innerHTML = "";
  }

  function renderSuccess(data) {
    statusArea.innerHTML =
      '<div class="alert alert-ok" role="status">Dự báo thành công (HTTP 200).</div>';
    resultArea.innerHTML = renderResult(data);
  }

  /* ---------------- sự kiện ---------------- */
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    clearFieldErrors();
    clearAreas();
    if (!validateClient()) {
      statusArea.innerHTML =
        '<div class="alert alert-err" role="alert"><p><strong>Dữ liệu chưa hợp lệ.</strong> ' +
        'Vui lòng sửa các ô được đánh dấu đỏ bên trên.</p></div>';
      var firstBad = form.querySelector('[aria-invalid="true"]');
      if (firstBad) { firstBad.focus(); }
      return;
    }

    renderLoading();
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span>Đang dự báo…';

    fetch(API_FORECAST, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(buildBody())
    })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (data) {
          if (r.ok) { renderSuccess(data); }
          else { renderError(data, r.status); }
        });
      })
      .catch(function (err) {
        renderError({
          message: "Không kết nối được tới server (" + err.message +
            "). Kiểm tra uvicorn còn đang chạy không."
        }, "network");
      })
      .then(function () {
        submitBtn.disabled = false;
        submitBtn.textContent = "DỰ BÁO LƯU LƯỢNG";
      });
  });

  resetBtn.addEventListener("click", function () {
    form.reset();
    clearFieldErrors();
    clearAreas();
    $("date").value = DEFAULTS.date;
    $("time").value = DEFAULTS.time;
    $("temperature").value = DEFAULTS.temperature;
    $("clouds").value = DEFAULTS.clouds;
    $("rain").value = DEFAULTS.rain;
    $("snow").value = DEFAULTS.snow;
  });

  /* khởi tạo */
  $("date").value = DEFAULTS.date;
  $("time").value = DEFAULTS.time;
  $("temperature").value = DEFAULTS.temperature;
  $("clouds").value = DEFAULTS.clouds;
  $("rain").value = DEFAULTS.rain;
  $("snow").value = DEFAULTS.snow;
  loadWeatherCategories();
})();
