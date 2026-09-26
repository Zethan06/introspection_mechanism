// Lazy Plotly renderer: render each 3-D scene only while it is on screen and
// purge its WebGL context once it scrolls away, so galleries with many scenes
// never exceed the browser's ~16 simultaneous WebGL-context limit (which would
// otherwise blank the earliest plots).
(function () {
  function specFor(id) {
    var s = document.querySelector('script.plot-spec[data-target="' + id + '"]');
    if (!s) return null;
    try { return JSON.parse(s.textContent); } catch (e) { return null; }
  }

  // Point filters are independent of the Plotly traces: Python keeps the
  // original color/marker grouping and records one optional filter id beside
  // each internal trace. A null id is a reference trace that is always shown.
  var activePointFilters = new Set();
  var pointSizeScale = 0.6;

  function readPointFilters() {
    activePointFilters.clear();
    var inputs = document.querySelectorAll("input[data-point-filter]");
    var allInput = document.querySelector("input[data-point-filter-all]");
    if (allInput && allInput.checked) {
      inputs.forEach(function (input) {
        activePointFilters.add(input.dataset.pointFilter);
      });
      return;
    }
    inputs.forEach(function (input) {
      if (input.checked) activePointFilters.add(input.dataset.pointFilter);
    });
  }

  function filteredData(spec) {
    if (!spec.pointFilters) return spec.data;
    return spec.data.map(function (trace, traceIndex) {
      var category = spec.pointFilters[traceIndex];
      if (category === null || category === undefined) return trace;
      var filtered = Object.assign({}, trace);
      filtered.visible = activePointFilters.has(category);
      return filtered;
    });
  }

  function scaledMarkerSize(size) {
    if (Array.isArray(size)) {
      return size.map(function (value) { return Number(value) * pointSizeScale; });
    }
    return Number(size) * pointSizeScale;
  }

  function plotData(spec) {
    return filteredData(spec).map(function (trace) {
      if (!trace.marker || trace.marker.size === undefined) return trace;
      var scaled = Object.assign({}, trace);
      scaled.marker = Object.assign({}, trace.marker);
      scaled.marker.size = scaledMarkerSize(trace.marker.size);
      return scaled;
    });
  }

  function displayConfig(spec) {
    var config = Object.assign({}, spec.config || {});
    config.toImageButtonOptions = {
      format: "png", width: 1200, height: 1200, scale: 2
    };
    return config;
  }

  function cleanDisplayAxis(axis) {
    var displayed = Object.assign({}, axis || {});
    displayed.visible = false;
    return displayed;
  }

  function cleanDisplayLayout(layout) {
    layout.scene = Object.assign({}, layout.scene || {});
    layout.scene.bgcolor = "rgba(0,0,0,0)";
    layout.scene.xaxis = cleanDisplayAxis(layout.scene.xaxis);
    layout.scene.yaxis = cleanDisplayAxis(layout.scene.yaxis);
    layout.scene.zaxis = cleanDisplayAxis(layout.scene.zaxis);
    layout.paper_bgcolor = "rgba(0,0,0,0)";
    layout.plot_bgcolor = "rgba(0,0,0,0)";
    return layout;
  }

  var jsPdfPromise = null;

  function ensureJsPdf() {
    if (window.jspdf) return Promise.resolve(window.jspdf);
    if (jsPdfPromise) return jsPdfPromise;
    jsPdfPromise = new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src =
        "https://cdn.jsdelivr.net/npm/jspdf@2.5.2/dist/jspdf.umd.min.js";
      script.onload = function () {
        if (window.jspdf) resolve(window.jspdf);
        else reject(new Error("PDF library loaded without a global export"));
      };
      script.onerror = function () {
        jsPdfPromise = null;
        reject(new Error("PDF library could not be loaded"));
      };
      document.head.appendChild(script);
    });
    return jsPdfPromise;
  }

  function numericValues(values) {
    if (Array.isArray(values)) return values.map(Number);
    if (ArrayBuffer.isView(values)) return Array.from(values, Number);
    if (!values || !values.bdata || !values.dtype) return [];
    var raw = window.atob(values.bdata);
    var buffer = new ArrayBuffer(raw.length);
    var bytes = new Uint8Array(buffer);
    for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    var constructors = {
      f4: Float32Array, f8: Float64Array,
      i1: Int8Array, u1: Uint8Array,
      i2: Int16Array, u2: Uint16Array,
      i4: Int32Array, u4: Uint32Array
    };
    var TypedArray = constructors[values.dtype];
    return TypedArray ? Array.from(new TypedArray(buffer), Number) : [];
  }

  function vector(x, y, z) { return { x: x, y: y, z: z }; }

  function subtract(a, b) {
    return vector(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

  function cross(a, b) {
    return vector(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x
    );
  }

  function normalize(v) {
    var length = Math.hypot(v.x, v.y, v.z) || 1;
    return vector(v.x / length, v.y / length, v.z / length);
  }

  function pdfRgb(color) {
    var match = /^#([0-9a-f]{6})$/i.exec(String(color || "#000000"));
    var rgbMatch = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i.exec(
      String(color || "")
    );
    var rgb = [0, 0, 0];
    if (match) {
      rgb = [
        parseInt(match[1].slice(0, 2), 16),
        parseInt(match[1].slice(2, 4), 16),
        parseInt(match[1].slice(4, 6), 16)
      ];
    } else if (rgbMatch) {
      rgb = rgbMatch.slice(1, 4).map(Number);
    }
    // PDF markers use the palette's original, fully saturated RGB values.
    // Pre-blending Plotly opacity with white made sparse points look washed out.
    return rgb;
  }

  function setPdfMarkerOpacity(pdf, opacity) {
    var alpha = Math.max(0, Math.min(1, Number(opacity)));
    if (!Number.isFinite(alpha)) alpha = 1;
    var key = alpha.toFixed(3);
    if (!pdf.__markerOpacityStates) pdf.__markerOpacityStates = {};
    if (!pdf.__markerOpacityStates[key]) {
      pdf.__markerOpacityStates[key] = new pdf.GState({ opacity: alpha });
    }
    pdf.setGState(pdf.__markerOpacityStates[key]);
  }

  function drawVectorMarker(pdf, point) {
    var rgb = pdfRgb(point.color);
    setPdfMarkerOpacity(pdf, point.opacity);
    pdf.setFillColor(rgb[0], rgb[1], rgb[2]);
    var radius = Number(point.radiusMm);
    var symbol = String(point.symbol || "circle");
    if (symbol.indexOf("diamond") !== -1) {
      pdf.lines(
        [[radius, radius], [-radius, radius], [-radius, -radius], [radius, -radius]],
        point.pageX,
        point.pageY - radius,
        [1, 1],
        "F",
        true
      );
    } else if (symbol.indexOf("square") !== -1) {
      pdf.rect(point.pageX - radius, point.pageY - radius, 2 * radius, 2 * radius, "F");
    } else if (symbol.indexOf("triangle") !== -1) {
      pdf.triangle(
        point.pageX,
        point.pageY - radius,
        point.pageX - radius,
        point.pageY + radius,
        point.pageX + radius,
        point.pageY + radius,
        "F"
      );
    } else {
      pdf.circle(point.pageX, point.pageY, radius, "F");
    }
  }

  function exportMillimetersPerPixel(source) {
    var plotSize = source._fullLayout && source._fullLayout._size;
    var width = plotSize && Number(plotSize.w) || source.clientWidth;
    var height = plotSize && Number(plotSize.h) || source.clientHeight;
    var displaySpan = Math.max(1, Math.min(width, height));
    return 180 / displaySpan;
  }

  function displayedAxisBounds(source, axisName, points) {
    var scene = source._fullLayout && source._fullLayout.scene;
    var axis = scene && scene[axisName + "axis"];
    var range = axis && axis.range;
    if (range && range.length === 2 && range.every(Number.isFinite)) {
      return { middle: (range[0] + range[1]) / 2, span: range[1] - range[0] || 1 };
    }
    var values = points.map(function (point) { return point[axisName]; });
    var minimum = Math.min.apply(null, values);
    var maximum = Math.max.apply(null, values);
    return { middle: (minimum + maximum) / 2, span: maximum - minimum || 1 };
  }

  function projectScenePoint(point, eye, forward, right, screenUp, focalDistance, perspective) {
    var fromEye = subtract(point, eye);
    var depth = dot(fromEye, forward);
    var scale = perspective ? focalDistance / Math.max(depth, 0.05) : 1;
    return {
      x: dot(fromEye, right) * scale,
      y: dot(fromEye, screenUp) * scale,
      depth: depth
    };
  }

  function vectorPoints(source, spec) {
    // Reuse the exact filtered and point-size-scaled traces used by render().
    // source.data is Plotly-owned and may contain the pre-transform input.
    var traces = plotData(spec).filter(function (trace) {
      return trace.type === "scatter3d" && trace.visible !== false &&
        trace.visible !== "legendonly";
    });
    var millimetersPerPixel = exportMillimetersPerPixel(source);
    var rawPoints = [];
    traces.forEach(function (trace) {
      var xs = numericValues(trace.x);
      var ys = numericValues(trace.y);
      var zs = numericValues(trace.z);
      var marker = trace.marker || {};
      var count = Math.min(xs.length, ys.length, zs.length);
      for (var i = 0; i < count; i++) {
        if (![xs[i], ys[i], zs[i]].every(Number.isFinite)) continue;
        rawPoints.push({
          x: xs[i], y: ys[i], z: zs[i],
          color: Array.isArray(marker.color) ? marker.color[i] : marker.color,
          opacity: Array.isArray(marker.opacity) ? marker.opacity[i] : marker.opacity,
          size: Array.isArray(marker.size) ? marker.size[i] : marker.size,
          symbol: Array.isArray(marker.symbol) ? marker.symbol[i] : marker.symbol,
          radiusMm: Math.max(
            0.05,
            Number(Array.isArray(marker.size) ? marker.size[i] : marker.size || 4) *
              millimetersPerPixel / 2
          )
        });
      }
    });
    if (!rawPoints.length) return [];

    var bounds = {};
    ["x", "y", "z"].forEach(function (axis) {
      bounds[axis] = displayedAxisBounds(source, axis, rawPoints);
    });
    var camera = source._fullLayout && source._fullLayout.scene &&
      source._fullLayout.scene.camera || defaultCamera();
    var eye = Object.assign(vector(1.25, 1.25, 1.25), camera.eye || {});
    var center = Object.assign(vector(0, 0, 0), camera.center || {});
    var up = Object.assign(vector(0, 0, 1), camera.up || {});
    var forward = normalize(subtract(center, eye));
    var right = normalize(cross(forward, up));
    var screenUp = normalize(cross(right, forward));
    var focalDistance = Math.hypot(
      eye.x - center.x, eye.y - center.y, eye.z - center.z
    ) || 1;
    var perspective = !camera.projection || camera.projection.type !== "orthographic";
    var projected = rawPoints.map(function (point) {
      // Plotly's scene cube is centered on zero with half-extent 0.5.
      var normalized = vector(
        (point.x - bounds.x.middle) / bounds.x.span,
        (point.y - bounds.y.middle) / bounds.y.span,
        (point.z - bounds.z.middle) / bounds.z.span
      );
      var screen = projectScenePoint(
        normalized, eye, forward, right, screenUp, focalDistance, perspective
      );
      return Object.assign({}, point, {
        projectedX: screen.x,
        projectedY: screen.y,
        depth: screen.depth
      });
    });

    // Fit the same fixed scene cube as Plotly, rather than enlarging the
    // visible point cloud to the page. This preserves point/cloud proportions.
    var projectedCube = [];
    [-0.5, 0.5].forEach(function (x) {
      [-0.5, 0.5].forEach(function (y) {
        [-0.5, 0.5].forEach(function (z) {
          projectedCube.push(projectScenePoint(
            vector(x, y, z), eye, forward, right, screenUp, focalDistance, perspective
          ));
        });
      });
    });
    var xValues = projectedCube.map(function (point) { return point.x; });
    var yValues = projectedCube.map(function (point) { return point.y; });
    var xMin = Math.min.apply(null, xValues);
    var xMax = Math.max.apply(null, xValues);
    var yMin = Math.min.apply(null, yValues);
    var yMax = Math.max.apply(null, yValues);
    var span = Math.max(xMax - xMin, yMax - yMin, 1e-9);
    var xMiddle = (xMin + xMax) / 2;
    var yMiddle = (yMin + yMax) / 2;
    projected.forEach(function (point) {
      point.pageX = 100 + (point.projectedX - xMiddle) * 180 / span;
      point.pageY = 100 - (point.projectedY - yMiddle) * 180 / span;
    });
    return projected.sort(function (a, b) { return b.depth - a.depth; });
  }

  async function exportPlotPdf(button) {
    var targetId = button.dataset.exportTarget;
    var source = document.getElementById(targetId);
    var spec = specFor(targetId);
    if (!source || !spec) throw new Error("plot data is unavailable");
    button.disabled = true;
    var oldLabel = button.textContent;
    button.textContent = "Exporting vector PDF…";
    try {
      await ensureJsPdf();
      var points = vectorPoints(source, spec);
      if (!points.length) throw new Error("the current view has no visible points");
      var pdf = new window.jspdf.jsPDF({
        orientation: "portrait", unit: "mm", format: [200, 200], compress: true
      });
      pdf.setProperties({
        title: (source.closest(".card") || {}).id || targetId,
        subject: "Vector export of the current 3D latent projection"
      });
      points.forEach(function (point) { drawVectorMarker(pdf, point); });
      var card = source.closest(".card");
      var filename = card && card.id ? card.id : targetId;
      pdf.save(filename + ".pdf");
    } finally {
      button.disabled = false;
      button.textContent = oldLabel;
    }
  }

  function refreshPointFilters() {
    readPointFilters();
    document.querySelectorAll(".lazy-plot").forEach(function (div) {
      if (div.dataset.rendered !== "1") return;
      var spec = specFor(div.id);
      if (!spec) return;
      cleanDisplayLayout(spec.layout);
      if (syncEnabled && lastCamera && spec.layout && spec.layout.scene) {
        spec.layout.scene.camera = copyCamera(lastCamera);
      }
      Plotly.react(div, plotData(spec), spec.layout, displayConfig(spec));
    });
  }

  // Synced 3-D camera: while enabled, rotating one rendered scene applies the
  // same eye/center/up vectors to every other currently-rendered scene.
  // Plotly fires plotly_relayout many times per frame during a drag, so the
  // actual cross-scene relayout is coalesced to once per animation frame --
  // applying it synchronously on every event would stack up dozens of
  // WebGL relayouts per drag gesture and freeze the page.
  var syncEnabled = true;
  var lastCamera = null;
  var pendingSource = null;
  var syncScheduled = false;
  var expectedCameras = new WeakMap();
  var syncingDivs = new WeakSet();
  var azimuthInput = null;
  var elevationInput = null;
  var cameraRadius = Math.sqrt(3 * 1.25 * 1.25);

  function copyCamera(camera) {
    return JSON.parse(JSON.stringify(camera));
  }

  function normalizeAngle(degrees) {
    var normalized = ((degrees + 180) % 360 + 360) % 360 - 180;
    return Math.abs(normalized) < 0.05 ? 0 : normalized;
  }

  function cameraAngles(camera) {
    var eye = camera.eye || {};
    var center = camera.center || {};
    var dx = Number(eye.x || 0) - Number(center.x || 0);
    var dy = Number(eye.y || 0) - Number(center.y || 0);
    var dz = Number(eye.z || 0) - Number(center.z || 0);
    var horizontal = Math.hypot(dx, dy);
    return {
      azimuth: horizontal < 1e-9
        ? 0
        : normalizeAngle(Math.atan2(dy, dx) * 180 / Math.PI),
      elevation: Math.atan2(dz, horizontal) * 180 / Math.PI
    };
  }

  function showCameraAngles(camera) {
    if (!azimuthInput || !elevationInput) return;
    var angles = cameraAngles(camera);
    azimuthInput.value = String(Math.round(angles.azimuth * 10) / 10);
    elevationInput.value = String(Math.round(angles.elevation * 10) / 10);
  }

  function defaultCamera() {
    return {
      eye: { x: 1.25, y: 1.25, z: 1.25 },
      center: { x: 0, y: 0, z: 0 },
      up: { x: 0, y: 0, z: 1 },
      projection: { type: "perspective" }
    };
  }

  function cameraWithAngles(azimuth, elevation) {
    var rotated = defaultCamera();
    var azimuthRadians = azimuth * Math.PI / 180;
    var elevationRadians = elevation * Math.PI / 180;
    var horizontal = cameraRadius * Math.cos(elevationRadians);
    rotated.eye = {
      x: horizontal * Math.cos(azimuthRadians),
      y: horizontal * Math.sin(azimuthRadians),
      z: cameraRadius * Math.sin(elevationRadians)
    };
    return rotated;
  }

  function applySyncedCamera(div, camera) {
    // Plotly may normalize the camera (for example by adding projection
    // defaults), so comparing the echoed object byte-for-byte is not enough
    // to prevent a feedback loop. Suppress all camera events briefly while a
    // programmatic relayout is settling.
    var expected = { until: Date.now() + 500 };
    expectedCameras.set(div, expected);
    syncingDivs.add(div);
    var relayout;
    try {
      relayout = Plotly.relayout(div, { "scene.camera": copyCamera(camera) });
    } catch (error) {
      syncingDivs.delete(div);
      throw error;
    }
    Promise.resolve(relayout).then(function () {
      window.requestAnimationFrame(function () { syncingDivs.delete(div); });
    }, function () {
      syncingDivs.delete(div);
      expectedCameras.delete(div);
    });
    window.setTimeout(function () {
      if (expectedCameras.get(div) === expected) expectedCameras.delete(div);
    }, 500);
  }

  function flushSync() {
    syncScheduled = false;
    if (!syncEnabled || !lastCamera) return;
    var camera = copyCamera(lastCamera);
    var divs = document.querySelectorAll(".lazy-plot");
    for (var i = 0; i < divs.length; i++) {
      var div = divs[i];
      if (div === pendingSource || div.dataset.rendered !== "1") continue;
      // Plotly emits plotly_relayout for programmatic relayouts too. Record the
      // camera before applying it so that event is recognized as an echo,
      // rather than becoming a new source and bouncing between every view.
      applySyncedCamera(div, camera);
    }
  }

  function syncCamera(sourceDiv, camera) {
    lastCamera = copyCamera(camera);
    showCameraAngles(lastCamera);
    pendingSource = sourceDiv;
    if (!syncEnabled || syncScheduled) return;
    syncScheduled = true;
    window.requestAnimationFrame(flushSync);
  }

  function setAllCameraAngles(azimuth, elevation) {
    if (!Number.isFinite(azimuth) || !Number.isFinite(elevation)) return;
    elevation = Math.max(-89.9, Math.min(89.9, elevation));
    lastCamera = cameraWithAngles(normalizeAngle(azimuth), elevation);
    showCameraAngles(lastCamera);
    pendingSource = null;
    document.querySelectorAll(".lazy-plot").forEach(function (div) {
      if (div.dataset.rendered === "1") applySyncedCamera(div, lastCamera);
    });
  }

  function render(div) {
    if (div.dataset.rendered === "1") return;
    var spec = specFor(div.id);
    if (!spec) return;
    cleanDisplayLayout(spec.layout);
    if (syncEnabled && lastCamera && spec.layout && spec.layout.scene) {
      spec.layout.scene.camera = copyCamera(lastCamera);
    }
    Plotly.newPlot(div, plotData(spec), spec.layout, displayConfig(spec));
    div.dataset.rendered = "1";
    div.on("plotly_relayout", function (update) {
      if (!update) return;
      var camera = update["scene.camera"];
      if (!camera) return;
      if (syncingDivs.has(div)) return;
      var expectedCamera = expectedCameras.get(div);
      if (expectedCamera && Date.now() <= expectedCamera.until) {
        return;
      }
      if (expectedCamera) expectedCameras.delete(div);
      syncCamera(div, camera);
    });
  }

  function purge(div) {
    if (div.dataset.rendered !== "1") return;
    Plotly.purge(div);
    expectedCameras.delete(div);
    div.dataset.rendered = "0";
  }

  function start() {
    var divs = Array.prototype.slice.call(
      document.querySelectorAll(".lazy-plot")
    );
    if (!divs.length) return;
    document.querySelectorAll(
      "input[data-point-filter], input[data-point-filter-all]"
    ).forEach(function (input) {
      input.addEventListener("change", refreshPointFilters);
    });
    readPointFilters();
    var pointSizeInput = document.getElementById("point-size");
    var pointSizeValue = document.getElementById("point-size-value");
    function applyPointSize() {
      var percentage = Number(pointSizeInput.value);
      if (!Number.isFinite(percentage)) return;
      percentage = Math.max(20, Math.min(150, percentage));
      pointSizeScale = percentage / 100;
      if (pointSizeValue) pointSizeValue.textContent = percentage + "%";
      refreshPointFilters();
    }
    if (pointSizeInput) {
      applyPointSize();
      pointSizeInput.addEventListener("input", applyPointSize);
    }
    divs.forEach(function (div) {
      var card = div.closest(".card");
      var metrics = card && card.querySelector(".metrics");
      if (!metrics || metrics.querySelector('[data-export-target="' + div.id + '"]')) {
        return;
      }
      var button = document.createElement("button");
      button.type = "button";
      button.className = "export-plot";
      button.dataset.exportTarget = div.id;
      button.textContent = "Export Vector PDF";
      metrics.appendChild(button);
    });
    document.querySelectorAll("button.export-plot").forEach(function (button) {
      button.addEventListener("click", function () {
        exportPlotPdf(button).catch(function (error) {
          window.alert("PDF export failed: " + error.message);
        });
      });
    });
    // rootMargin pre-renders one viewport ahead / keeps one behind for smooth
    // scrolling while capping concurrent contexts well under the limit.
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) render(e.target);
        else purge(e.target);
      });
    }, { root: null, rootMargin: "300px 0px 300px 0px", threshold: 0.01 });
    divs.forEach(function (d) { io.observe(d); });

    var toggle = document.getElementById("sync-rotation");
    if (toggle) {
      syncEnabled = toggle.checked;
      toggle.addEventListener("change", function () {
        syncEnabled = toggle.checked;
        if (syncEnabled && lastCamera) syncCamera(null, lastCamera);
      });
    }
    azimuthInput = document.getElementById("camera-azimuth");
    elevationInput = document.getElementById("camera-elevation");
    function applyAngleInputs() {
      setAllCameraAngles(
        Number(azimuthInput.value),
        Number(elevationInput.value)
      );
    }
    if (azimuthInput && elevationInput) {
      azimuthInput.addEventListener("change", applyAngleInputs);
      elevationInput.addEventListener("change", applyAngleInputs);
    }
  }

  if (window.Plotly) {
    start();
  } else {
    // Plotly is loaded by an earlier script tag; wait for it.
    var wait = setInterval(function () {
      if (window.Plotly) { clearInterval(wait); start(); }
    }, 50);
  }
})();
