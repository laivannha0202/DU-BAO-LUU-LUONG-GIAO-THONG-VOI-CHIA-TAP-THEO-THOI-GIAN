/* ==========================================================================
   app/static/js/dashboard.js
   Màn 3 — Dashboard + Model Card.
   NGUYÊN TẮC: KHÔNG hard-code bất kỳ metric nào trong file này.
   Mọi con số đến từ GET /api/dashboard-metrics (đọc artifact thật).
   ========================================================================== */
(function () {
  "use strict";

  var API = "/api/dashboard-metrics";
  var API_INFO = "/api/model-info";

  var $ = function (id) { return document.getElementById(id); };
  var DOW = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"];

  /* ---------------- tiện ích ---------------- */
  function esc(t) {
    return String(t == null ? "" : t)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function n(v, d) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    d = d === undefined ? 2 : d;
    return Number(v).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function i(v) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    return Number(v).toLocaleString("vi-VN");
  }
  function table(el, caption, head, rows, highlightRow) {
    var html = "<caption>" + esc(caption) + "</caption><thead><tr>" +
      head.map(function (h, k) {
        return "<th" + (k === 0 ? ' scope="col"' : ' class="num" scope="col"') + ">" + esc(h) + "</th>";
      }).join("") + "</tr></thead><tbody>" +
      rows.map(function (r, idx) {
        var trClass = (highlightRow != null && idx === highlightRow) ? ' class="t-highlight"' : "";
        return "<tr" + trClass + ">" + r.map(function (c, k) {
          var tag = k === 0 ? "th scope=\"row\"" : "td class=\"num\"";
          return "<" + tag + ">" + (c && c.html ? c.html : esc(c)) + "</" + tag.split(" ")[0] + ">";
        }).join("") + "</tr>";
      }).join("") + "</tbody>";
    el.innerHTML = html;
  }
  function metric(label, value, sub, cls) {
    return '<div class="metric ' + (cls || "") + '">' +
      '<div class="label">' + esc(label) + "</div>" +
      '<div class="value">' + esc(value) + "</div>" +
      '<div class="sub">' + esc(sub || "") + "</div></div>";
  }
  /* Biểu đồ cột GHÉP: mỗi nhóm có N chuỗi (series) cạnh nhau. */
  function groupedBars(el, axisEl, labels, series) {
    var max = 1;
    labels.forEach(function (_, k) {
      series.forEach(function (s) {
        var v = s.values[k];
        if (v != null && !isNaN(v) && v > max) { max = v; }
      });
    });
    el.innerHTML = labels.map(function (label, k) {
      return series.map(function (s) {
        var v = s.values[k];
        var pct = (v == null || isNaN(v)) ? 0 : Math.max(2, (v / max) * 100);
        return '<div class="bar ' + (s.cls || "") + '" style="height:' + pct + '%" title="' +
          esc(label) + " — " + esc(s.name) + ": " + esc(v) + '"></div>';
      }).join("");
    }).join("");
    axisEl.innerHTML = labels.map(function (l) { return "<span>" + esc(l) + "</span>"; }).join("");
  }

  /* ---------------- render ---------------- */
  function render(d) {
    var ft = d.final_test || {};
    var ridge = ft.ridge || {};
    var base = ft.baseline || {};
    var val = (d.validation_reference || {}).ridge || {};
    var model = d.model || {};
    var hp = model.hyperparameters || {};
    var split = d.time_split || {};
    var ea = d.error_analysis || {};
    var ds = d.dataset || {};

    /* --- 1. FINAL TEST --- */
    var improve = (base.MAE != null && ridge.MAE != null)
      ? (base.MAE - ridge.MAE) : null;
    var improvePct = improve != null ? (improve / base.MAE) * 100 : null;

    $("final-metrics").innerHTML =
      metric("MAE — Ridge (FINAL TEST)", n(ridge.MAE), "xe/giờ · thấp hơn là tốt hơn", "ok") +
      metric("RMSE — Ridge", n(ridge.RMSE), "xe/giờ", "ok") +
      metric("R² — Ridge", n(ridge.R2, 4), "1 là hoàn hảo", "ok") +
      metric("MAE — Baseline", n(base.MAE), "trung bình theo giờ × thứ", "warn") +
      metric("Cải thiện MAE", improve != null ? n(improve) : "—",
        improvePct != null ? n(improvePct, 1) + "% so với baseline" : "", "ok") +
      metric("Số dòng test", i(ridge.n), "giờ trong năm 2018");

    table($("final-table"),
      "So sánh baseline và mô hình trên FINAL TEST 2018 (cùng một tập dữ liệu)",
      ["Mô hình", "MAE", "RMSE", "R²", "n"],
      [
        ["Baseline (mean theo giờ × thứ)", n(base.MAE), n(base.RMSE), n(base.R2, 4), i(base.n)],
        ["Ridge pipeline (đã đóng băng)", n(ridge.MAE), n(ridge.RMSE), n(ridge.R2, 4), i(ridge.n)]
      ], 1);

    $("final-note").textContent =
      "MAE trên FINAL TEST (" + n(ridge.MAE) + ") thấp hơn MAE trên VALIDATION (" +
      n(val.MAE) + ") — tức hiệu năng không suy giảm khi dữ liệu dài thêm một năm. " +
      "Cả hai đều là đánh giá out-of-sample trên các tập rời nhau theo thời gian.";

    /* --- 2. TIME SPLIT --- */
    var names = { train: "TRAIN (huấn luyện)", validation: "VALIDATION (chọn alpha)",
                  test: "FINAL TEST (chỉ đánh giá)" };
    table($("split-table"), "Ba tập được chia theo thời gian, không xáo trộn",
      ["Tập", "Khoảng thời gian", "Số dòng"],
      Object.keys(names).map(function (k) {
        var s = split[k] || {};
        var range = [s.start, s.end].filter(Boolean).join(" → ");
        return [names[k], range, i(s.n_rows)];
      }));

    var a = (split.assertions) || {};
    $("split-assert").textContent =
      "Assert: max(train) < min(validation) = " + a.max_train_lt_min_validation +
      " · max(validation) < min(test) = " + a.max_validation_lt_min_test +
      " · không có timestamp chung = " + a.no_shared_timestamps + ".";

    /* --- 3. ALPHA --- */
    var tuning = d.alpha_tuning || {};
    var results = tuning.results || [];
    var best = tuning.best_alpha;
    var trows = results.map(function (r) {
      return [String(r.alpha), n(r.MAE), n(r.RMSE), n(r.R2, 4)];
    });
    var bestIdx = results.findIndex(function (r) { return r.alpha === best; });
    table($("alpha-table"),
      "Lưới alpha — chọn theo " + (tuning.criterion || "validation MAE") +
      ". Alpha được chọn: " + (best === undefined ? "—" : String(best)),
      ["alpha", "MAE (validation)", "RMSE (validation)", "R² (validation)"],
      trows, bestIdx);

    /* --- 4. BY HOUR + actual vs predicted --- */
    var hourSegs = ((ea.hour) || {}).segments || [];
    if (hourSegs.length) {
      var labels = hourSegs.map(function (s) { return String(s.segment).replace("hour_", ""); });
      groupedBars($("avp-chart"), $("avp-axis"), labels, [
        { name: "Lưu lượng thực (mean_actual)", cls: "", values: hourSegs.map(function (s) { return s.mean_actual; }) },
        { name: "Lưu lượng dự báo (mean_pred)", cls: "b2", values: hourSegs.map(function (s) { return s.mean_pred; }) }
      ]);
      groupedBars($("hour-chart"), $("hour-axis"), labels, [
        { name: "MAE Ridge", cls: "", values: hourSegs.map(function (s) { return s.MAE; }) },
        { name: "MAE baseline", cls: "b2", values: hourSegs.map(function (s) { return s.baseline_MAE; }) }
      ]);

      table($("hour-table"), "MAE / RMSE / bias theo giờ trong ngày (FINAL TEST 2018)",
        ["Giờ", "n mẫu", "MAE Ridge", "MAE baseline", "RMSE", "Lưu lượng thực TB", "Dự báo TB", "Bias"],
        hourSegs.map(function (s) {
          return [
            String(s.segment).replace("hour_", "") + ":00", i(s.n_samples),
            n(s.MAE), n(s.baseline_MAE), n(s.RMSE),
            i(Math.round(s.mean_actual)), i(Math.round(s.mean_pred)), n(s.bias)
          ];
        }));
    }
    $("avp-note").textContent = (d.actual_vs_predicted || {}).note || "";

    /* --- 5. BY DAY OF WEEK --- */
    var dowSegs = ((ea.day_of_week) || {}).segments || [];
    if (dowSegs.length) {
      table($("dow-table"), "MAE theo ngày trong tuần (FINAL TEST 2018)",
        ["Thứ", "n mẫu", "MAE Ridge", "MAE baseline", "RMSE", "Lưu lượng thực TB", "Bias"],
        dowSegs.map(function (s) {
          return [DOW[s.day_of_week] || s.segment, i(s.n_samples), n(s.MAE), n(s.baseline_MAE),
            n(s.RMSE), i(Math.round(s.mean_actual)), n(s.bias)];
        }));
    }

    /* --- 6. HOLIDAY --- */
    var hol = ea.holiday || {};
    if ((hol.segments || []).length) {
      table($("holiday-table"),
        "Sai số ở ngày lễ — luôn kèm số mẫu vì nhóm ngày lễ rất nhỏ",
        ["Phân khúc", "n mẫu", "MAE Ridge", "MAE baseline", "RMSE", "Lưu lượng thực TB", "Bias"],
        hol.segments.map(function (s) {
          return [s.segment === "holiday" ? "Ngày lễ" : "Ngày thường", i(s.n_samples),
            n(s.MAE), n(s.baseline_MAE), n(s.RMSE), i(Math.round(s.mean_actual)), n(s.bias)];
        }));
      $("holiday-note").textContent =
        "Chênh lệch MAE giữa ngày lễ và ngày thường là " + n(hol.mae_delta) +
        " xe/giờ. Trong FINAL TEST 2018 chỉ có " + i(hol.n_holiday_dates) +
        " ngày lễ (n = " + i((hol.segments.find(function (s) { return s.segment === "holiday"; }) || {}).n_samples) +
        " giờ) → đây là ước lượng có độ bất định cao, không nên đọc như một con số chắc chắn.";
    }

    /* --- 7. WEATHER + EXTREME --- */
    var wmSegs = ((ea.weather_main) || {}).segments || [];
    if (wmSegs.length) {
      table($("weather-table"), "MAE theo hiện tượng thời tiết (multi-hot: các phân khúc có thể trùng nhau)",
        ["Thời tiết", "n mẫu", "MAE Ridge", "MAE baseline", "RMSE", "Lưu lượng thực TB", "Bias", "Ghi chú"],
        wmSegs.map(function (s) {
          return [s.segment, i(s.n_samples), n(s.MAE), n(s.baseline_MAE), n(s.RMSE),
            i(Math.round(s.mean_actual)), n(s.bias), s.note || (s.small_sample ? "Mẫu nhỏ" : "—")];
        }));
    }
    var ex = ea.extreme_weather || {};
    if ((ex.segments || []).length) {
      table($("extreme-table"), "MAE theo nhóm thời tiết cực đoan",
        ["Phân khúc", "n mẫu", "MAE Ridge", "MAE baseline", "RMSE", "Bias", "Ghi chú"],
        ex.segments.map(function (s) {
          return [s.segment, i(s.n_samples), n(s.MAE), n(s.baseline_MAE), n(s.RMSE), n(s.bias),
            s.note || (s.small_sample ? "Mẫu nhỏ" : "—")];
        }));
      $("extreme-note").textContent = ex.multi_label_note || "";
      $("weather-note").textContent =
        "Mỗi phân khúc đều kèm số mẫu. Phân khúc không có mẫu được ghi rõ “không đánh giá được” " +
        "thay vì báo số 0.";
    }

    /* --- 8. FIGURES (đường dẫn lấy từ artifact) --- */
    var figs = d.figures || {};
    var grid = $("figures-grid");
    grid.innerHTML = "";
    Object.keys(figs).forEach(function (key) {
      var src = String(figs[key]).replace(/^reports\/figures\//, "/figures/");
      var fig = document.createElement("figure");
      fig.innerHTML = '<img src="' + esc(src) + '" alt="Biểu đồ ' + esc(key) +
        '" loading="lazy"><figcaption>' + esc(key) + " — sinh bởi <code>src/evaluate.py</code> " +
        "từ FINAL TEST 2018.</figcaption>";
      grid.appendChild(fig);
    });
    if (!grid.children.length) {
      grid.innerHTML = '<p class="small muted">Artifact không có đường dẫn hình ảnh nào.</p>';
    }

    /* --- 9. EXPERIMENTS --- */
    var ex9 = d.experiments || {};
    var e1 = ex9.experiment_1_random_vs_time_split || {};
    if (e1.time_split && (e1.per_seed || []).length) {
      var s1 = e1.summary || {};
      var same = s1.delta_mae_same_rows || {};
      var diff = s1.delta_mae_different_test_sets || {};
      table($("exp1-table"),
        "Random split vs time split — lặp " + (e1.per_seed.length) +
        " seed cố định, báo cáo trung bình ± độ lệch chuẩn",
        ["Phép đo", "MAE trung bình ± SD", "min", "max"],
        [
          ["(a) Mỗi arm dùng tập test riêng — " +
            n(diff.mean) + " ± " + n(diff.sd),
            n(diff.mean), n(diff.sd), n(diff.min) + " … " + n(diff.max)],
          ["(b) CÙNG một tập dòng đánh giá — " +
            n(same.mean) + " ± " + n(same.sd),
            n(same.mean), n(same.sd), n(same.min) + " … " + n(same.max)]
        ]);
      var worst = e1.per_seed.reduce(function (acc, r) {
        return r.random_split.MAE < acc ? r.random_split.MAE : acc;
      }, Infinity);
      var best = e1.per_seed.reduce(function (acc, r) {
        return r.random_split.MAE > acc ? r.random_split.MAE : acc;
      }, 0);
      $("exp1-note").textContent =
        "Time split (test " + (e1.time_split.test_range || []).map(function (d) {
          return d.slice(0, 10);
        }).join(" → ") + ", n = " + i(e1.time_split.n) + "): MAE " + n(e1.time_split.MAE) +
        ". MAE random split dao động " + n(worst) + " … " + n(best) +
        " theo seed. " + (e1.caveat || "");
    }

    var e1b = ex9.experiment_1b_leakage_controlled || {};
    var runs1b = e1b.per_seed || [];
    if (runs1b.length) {
      var labels = e1b.arm_labels || {};
      var keys = Object.keys(labels);
      var firstRun = runs1b[0];
      var sd = function (key) {
        var vals = runs1b.map(function (r) { return r.arms[key].MAE; });
        var mean = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
        var varr = vals.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / vals.length;
        return { mean: mean, sd: Math.sqrt(varr) };
      };
      var nmed = (e1b.summary || {}).n_train_median || {};
      var n2017 = (e1b.summary || {}).n_train_rows_from_2017_median || {};
      table($("exp1b-table"),
        "4 arm dùng CHUNG một tập test (n = " + i(firstRun.shared_test_n) +
        " giờ, seed đầu " + firstRun.seed + ") — arm D là arm đối chứng CÙNG KÍCH THƯỚC với B",
        ["Arm", "Mô tả", "n train", "Dòng từ 2017", "MAE (TB ± SD)", "RMSE", "R²"],
        keys.map(function (k) {
          var a = firstRun.arms[k];
          var st = sd(k);
          return [k.split("_")[0], (labels[k] || "").split(": ").slice(1).join(": "),
            i(nmed[k] != null ? nmed[k] : a.n_train), i(n2017[k] != null ? n2017[k] : 0),
            n(st.mean) + " ± " + n(st.sd), n(a.RMSE), n(a.R2, 4)];
        }));
      $("exp1b-note").textContent = (e1b.interpretation || "");
    }

    var e1c = ex9.experiment_1c_block_neighbour || {};
    if (e1c.arms) {
      var cKeys = Object.keys(e1c.arms);
      table($("exp1c-table"),
        "Thí nghiệm 1c — khối liên tục: tháng chẵn của 2017 vào train, tháng lẻ làm test chung" +
        " (n = " + i((e1c.shared_test || {}).n) + ")",
        ["Arm", "n train", "Dòng từ 2017", "MAE", "RMSE", "R²"],
        cKeys.map(function (k) {
          var a = e1c.arms[k];
          return [k.split("_")[0], i(a.n_train), i(a.n_train_rows_from_2017),
            n(a.MAE), n(a.RMSE), n(a.R2, 4)];
        }));
      $("exp1c-note").textContent =
        "ΔMAE (P2 − P1) = " + n(e1c.delta_MAE_P2_minus_P1) + ". " + (e1c.interpretation || "");
    }

    var e3 = ex9.experiment_3_rolling_origin || {};
    if ((e3.folds || []).length) {
      table($("exp3-table"), "Rolling origin — mọi fold đều out-of-sample",
        ["Fold", "Năm test", "Khoảng train", "Khoảng test", "n", "MAE", "RMSE", "R²"],
        e3.folds.map(function (f) {
          return [String(f.fold), String(f.test_year), (f.train_range || []).join(" → "),
            (f.test_range || []).join(" → "), i(f.n), n(f.MAE), n(f.RMSE), n(f.R2, 4)];
        }));
      var tr = e3.mae_trend || {};
      var pooled = e3.pooled_out_of_sample || {};
      $("exp3-note").textContent =
        (tr.verdict || "") + " Tổng hợp " + i(pooled.n) + " giờ out-of-sample trong " +
        (pooled.years || []).join(", ") + ": MAE = " + n(pooled.MAE) + ".";
    }

    var e3b = ex9.experiment_3b_covariate_drift || {};
    var psi = e3b.psi_vs_reference || {};
    var th = e3b.psi_thresholds || {};
    if (Object.keys(psi).length) {
      var cols = Object.keys(psi);
      var years = Object.keys(psi[cols[0]]).sort();
      table($("psi-table"),
        "PSI so với năm tham chiếu " + e3b.reference_year +
        " (ổn định < " + th.stable + " · trung bình " + th.stable + "–" + th.moderate +
        " · dịch chuyển mạnh > " + th.moderate + ")",
        ["Biến"].concat(years),
        cols.map(function (c) {
          return [c].concat(years.map(function (y) {
            var v = (psi[c] || {})[y];
            return { html: v == null ? "—" : "<span class=\"badge\">" + n(v, 4) + "</span>" };
          }));
        }));
      $("psi-note").textContent =
        "Đây là kiểm tra dịch chuyển PHÂN BỐ của biến đầu vào, tách bạch khỏi phép so sánh " +
        "chất lượng dự báo theo tập (mục 9c).";
    }

    /* --- 10. MODEL CARD --- */
    var fit = model.fit_data || {};
    table($("card-summary"), "Thông tin mô hình",
      ["Mục", "Giá trị"],
      [
        ["Loại mô hình", "Hồi quy tuyến tính L2 — Ridge Regression (sklearn.linear_model.Ridge)"],
        ["Hyperparameter", "alpha = " + (hp.alpha !== undefined ? String(hp.alpha) : "—") +
          " · solver = " + (hp.solver || "—")],
        ["Tiêu chí chọn alpha", (model.alpha_tuning || {}).criterion || "—"],
        ["Target", "<code>traffic_volume</code> — lưu lượng, đơn vị xe/giờ"],
        ["Số lượng đặc trưng sau tiền xử lý", i(model.n_features_out)],
        ["Artifact phục vụ", "<code>models/ridge_pipeline.joblib</code>"],
        ["Phạm vi huấn luyện", (fit.start || "—") + " → " + (fit.end || "—") +
          " (" + i(fit.n_rows) + " dòng)"]
      ]);

    var cov = ds.coverage || {};
    table($("card-data"), "Dữ liệu",
      ["Mục", "Giá trị"],
      [
        ["Nguồn", "UCI Machine Learning Repository — Metro Interstate Traffic Volume (CC BY 4.0)"],
        ["Vị trí", "I-94 westbound, trạm ATR 301"],
        ["Số dòng thô", i(ds.raw_rows)],
        ["Số dòng dùng để mô hình hoá (sau khi collapse trùng timestamp)",
          i(ds.modeling_rows)],
        ["Số dòng bị loại do trùng timestamp", i(ds.rows_removed_by_collapse)],
        ["Khoảng thời gian quan sát", (cov.start || "—") + " → " + (cov.end || "—")],
        ["Đơn vị quan sát", "1 giờ tại một trạm đo"],
        ["Đơn vị mục tiêu", "xe/giờ (traffic_volume)"]
      ]);

    var fc = model.feature_columns || {};
    var pre = model.preprocessing || {};
    table($("card-features"), "Đặc trưng & tiền xử lý",
      ["Nhóm", "Cột", "Tiền xử lý (fit trên TRAIN duy nhất)"],
      [
        ["Danh mục", (fc.categorical || []).join(", "), (pre.categorical || []).join(" → ") || "—"],
        ["Số", (fc.numeric || []).join(", "), (pre.numeric || []).join(" → ") || "—"],
        ["Nhị phân", (fc.binary || []).join(", "), (pre.binary || []).join(" ") || "—"]
      ]);

    table($("card-metrics"), "Số liệu đánh giá (tất cả đều out-of-sample)",
      ["Tập / mô hình", "MAE", "RMSE", "R²", "n", "Vai trò"],
      [
        ["VALIDATION — Ridge", n(val.MAE), n(val.RMSE), n(val.R2, 4), i(val.n),
          "dùng để chọn alpha"],
        ["FINAL TEST — Ridge", n(ridge.MAE), n(ridge.RMSE), n(ridge.R2, 4), i(ridge.n),
          "kết quả chính thức"],
        ["FINAL TEST — Baseline", n(base.MAE), n(base.RMSE), n(base.R2, 4), i(base.n),
          "đường tham chiếu"]
      ], 1);

    /* Hạn chế: sinh từ artifact + phần định tính đã được nhóm thống nhất */
    var holSeg = ((ea.holiday || {}).segments || []).find(function (s) { return s.segment === "holiday"; });
    var exSeg = ((ea.extreme_weather || {}).segments || []).find(function (s) { return s.segment === "any_extreme_weather"; });
    var normalSeg = ((ea.extreme_weather || {}).segments || []).find(function (s) { return s.segment === "no_extreme_weather"; });
    var limits = [
      "Chỉ dự đoán cho <strong>một trạm đo duy nhất</strong> (ATR 301, I-94 chiều westbound). " +
      "Kết quả không đại diện cho toàn thành phố và không suy rộng được cho trạm hoặc hướng khác.",
      "Sai số cao hơn nhiều ở <strong>ngày lễ</strong>: MAE = " + n(holSeg && holSeg.MAE) +
      " xe/giờ với n = " + i(holSeg && holSeg.n_samples) + " giờ, so với " +
      n((ea.holiday || {}).segments && ((ea.holiday || {}).segments[0] || {}).MAE) +
      " ở ngày thường. Nhóm ngày lễ rất nhỏ nên ước lượng có độ bất định lớn.",
      "Sai số cao hơn ở <strong>thời tiết cực đoan</strong>: MAE = " + n(exSeg && exSeg.MAE) +
      " (n = " + i(exSeg && exSeg.n_samples) + ") so với " + n(normalSeg && normalSeg.MAE) +
      " khi không có hiện tượng cực đoan (n = " + i(normalSeg && normalSeg.n_samples) + ").",
      "Tập kiểm định cuối 2018 chỉ kéo dài <strong>tới 30/09/2018</strong> — không có dữ liệu cho " +
      "tháng 10–12 năm 2018 để đánh giá.",
      "Dữ liệu là <strong>lịch sử 2012–2018</strong>. Mô hình không cập nhật theo thay đổi hạ tầng, " +
      "chính sách giao thông, giá nhiên liệu hay hành vi người dùng sau thời điểm này.",
      "Ngày lễ <em>State Fair</em> không có quy tắc lịch, được lấy từ bảng ngày công bố chỉ cho " +
      "một khoảng năm hữu hạn. Ngoài khoảng đó hệ thống <strong>báo rõ thay vì tự đoán</strong>.",
      "<strong>Không dùng cho mục đích safety-critical</strong>: không dùng để điều khiển giao thông, " +
      "điều khiển tín hiệu hay cảnh báo an toàn.",
      "Mô hình tuyến tính: không nắm được hiệu ứng phi tuyến phức tạp, và dự báo có thể lệch đáng kể " +
      "ở các tình huống chưa từng xuất hiện trong dữ liệu huấn luyện.",
      "Ở <strong>giờ đêm</strong> (00–04, 22–23) baseline lại tốt hơn Ridge: Ridge không phải lựa chọn " +
      "tối ưu cho mọi khung giờ.",
      "Ridge có thể trả dự báo <strong>âm</strong> vì là hồi quy tuyến tính. Ứng dụng chặn về 0 " +
      "(<code>max(0, ·)</code>), nhưng đó là policy tạm chứ không phải sửa gốc vấn đề."
    ];
    $("card-limitations").innerHTML = limits.map(function (x) { return "<li>" + x + "</li>"; }).join("");

    /* Hậu xử lý max(0, ·) — số liệu từ reports/figures/postprocess_audit.json */
    renderPostprocess(d.serving_policy, d.postprocess_audit);

    $("dash").hidden = false;
    $("load-status").innerHTML = "";
  }

  /* Chính sách phục vụ đã đóng băng + báo cáo tác động — LẤY TỪ ARTIFACT, không gõ tay. */
  function renderPostprocess(policy, audit) {
    var box = $("postprocess-note");
    if (!box) { return; }

    if (!policy || !policy.policy_artifact_present) {
      box.innerHTML = '<div class="card"><h3 class="mt0">Hậu xử lý kết quả âm</h3>' +
        '<div class="alert alert-warn"><p class="mb0">Chưa có artifact ' +
        "<code>models/serving_policy.json</code>. Chạy " +
        "<code>py src\freeze_serving_policy.py</code> để chốt policy trên TRAIN + " +
        "VALIDATION và sinh bằng chứng.</p></div></div>";
      return;
    }

    var ev = (policy.validation_evidence || {});
    var html = '<div class="card"><h3 class="mt0">Hậu xử lý kết quả âm — chính sách đã đóng băng</h3>' +
      '<div class="alert alert-info"><p><strong>RAW MODEL</strong> = Ridge(alpha, lsqr) trả về ' +
      "trực tiếp. <strong>DEPLOYED PREDICTOR</strong> = <code>max(0, ·)</code> ∘ Ridge — một " +
      "<em>wrapper phục vụ</em>, KHÔNG phải mô hình khác. Hai hàng số dưới đây " +
      "<strong>không phải cùng một model metric</strong> và không được gọi chung tên.</p></div>";

    // --- Quyết định ---
    html += '<div class="table-wrap"><table>' +
      "<caption>Quyết định policy — chốt trên TRAIN + VALIDATION, KHÔNG dùng FINAL TEST 2018</caption>" +
      "<thead><tr><th scope=\"col\">Mục</th><th scope=\"col\">Giá trị</th></tr></thead><tbody>" +
      "<tr><th scope=\"row\">Chính sách</th><td><code>" + esc(policy.policy_id) + "</code> " +
        (policy.version ? "(" + esc(policy.version) + ")" : "") + "</td></tr>" +
      "<tr><th scope=\"row\">Quy tắc</th><td><code>" + esc(policy.rule) + "</code></td></tr>" +
      "<tr><th scope=\"row\">Chốt trên</th><td>" +
        esc((policy.selected_on || []).join(" · ")) + "</td></tr>" +
      "<tr class=\"t-highlight\"><th scope=\"row\">FINAL TEST dùng để chọn policy?</th>" +
        "<td><strong>" + (policy.final_test_used_for_selection ? "CÓ" : "KHÔNG") +
        "</strong></td></tr>" +
      "<tr><th scope=\"row\">Căn cứ (a) — miền giá trị</th><td>" +
        esc(policy.justification_domain) + " Sàn miền giá trị từ TRAIN = " +
        n(policy.domain_floor_from_train) + "</td></tr>" +
      "<tr><th scope=\"row\">Căn cứ (b) — VALIDATION 2017</th><td>" +
        esc(policy.justification_validation) + "</td></tr>" +
      "</tbody></table></div>";

    // --- Bằng chứng validation ---
    if (ev.n_raw_negative !== undefined && ev.n_raw_negative !== null) {
      html += '<h4 class="mt0" style="margin-top:1rem">Bằng chứng định lượng trên VALIDATION 2017 ' +
        "(n = " + n(ev.n_rows, 0) + ")</h4><ul class=\"small\">" +
        "<li>Số dự báo thô âm: <strong>" + n(ev.n_raw_negative, 0) + "</strong> (" +
          n(ev.share_raw_negative_pct, 2) + " %)</li>" +
        "<li>Dự báo thô nhỏ nhất: <strong>" + n(ev.min_raw_prediction) + "</strong></li>" +
        "<li>min(traffic_volume) thực tế: <strong>" + n(ev.min_actual_target) + "</strong> → sàn đúng là 0</li>" +
        "<li>Sai số tuyệt đối <strong>không tăng ở bất kỳ dòng nào</strong> khi cắt: " +
          (ev.pointwise_abs_error_never_increases ? "<strong>đúng</strong>" : "<strong>SAI</strong>") +
          " (số dòng đổi: " + n(ev.n_rows_where_clip_changes_error, 0) + ")</li>" +
        "</ul>";
    }

    html += '<p class="small muted">Lập luận toán học (không cần dữ liệu): ' +
      esc(policy.mathematical_argument || "—") + "</p>";

    // --- Báo cáo tác động sau khi đóng băng ---
    if (audit && (audit.splits || []).length) {
      html += '<h4 class="mt0" style="margin-top:1.2rem">Báo cáo tác động — chạy SAU khi đã đóng băng</h4>' +
        '<div class="table-wrap"><table>' +
        "<caption>Metric của RAW MODEL là kết luận chính thức. Cột DEPLOYED PREDICTOR là " +
        "wrapper phục vụ, báo riêng.</caption>" +
        "<thead><tr><th scope=\"col\">Tập</th><th scope=\"col\" class=\"num\">n âm (raw)</th>" +
        "<th scope=\"col\" class=\"num\">min raw</th>" +
        "<th scope=\"col\" class=\"num\">MAE — RAW MODEL</th>" +
        "<th scope=\"col\" class=\"num\">MAE — DEPLOYED</th>" +
        "<th scope=\"col\" class=\"num\">RMSE — RAW</th>" +
        "<th scope=\"col\" class=\"num\">RMSE — DEPLOYED</th></tr></thead><tbody>";
      audit.splits.forEach(function (s) {
        var rw = s.metrics_raw, dp = s.metrics_deployed;
        html += "<tr><th scope=\"row\">" + esc(s.split) + "</th>" +
          '<td class="num">' + n(s.n_raw_negative, 0) + " (" + n(s.share_raw_negative_pct, 2) + " %)</td>" +
          '<td class="num">' + n(s.min_raw_prediction) + "</td>" +
          '<td class="num"><strong>' + n(rw.MAE) + "</strong></td>" +
          '<td class="num">' + n(dp.MAE) + "</td>" +
          '<td class="num">' + n(rw.RMSE) + "</td>" +
          '<td class="num">' + n(dp.RMSE) + "</td></tr>";
      });
      html += "</tbody></table></div>";
    }

    html += '<p class="small muted mb0">Nhóm <strong>không</strong> chọn policy vì nhìn thấy số ' +
      "dự báo âm trên FINAL TEST 2018, và <strong>không</strong> sửa lại gói đánh giá đã đóng " +
      "băng để khớp API — làm vậy tức là dùng kết quả 2018 để điều chỉnh mô hình. " +
      "Chặn về sàn là <em>policy</em>, không phải <em>học</em>: lưu lượng thực ở những giờ đó " +
      "vẫn dương.</p></div>";
    box.innerHTML = html;
  }

  /* ---------------- load ---------------- */
  function fail(message, status) {
    $("load-status").innerHTML =
      '<div class="alert alert-err" role="alert"><p><strong>Không tải được số liệu' +
      (status ? " (HTTP " + esc(status) + ")" : "") + ".</strong> " + esc(message) + "</p></div>";
  }

  fetch(API, { headers: { "Accept": "application/json" } })
    .then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) {
          throw new Error((body && body.detail && body.detail.message) ||
            (body && body.message) || "Lỗi không xác định");
        }
        return body;
      });
    })
    .then(render)
    .catch(function (err) { fail(err.message); });
})();
