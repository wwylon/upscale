/** Upscale bridge — estimate. Attaches to window.__UpscaleBridge. */

(function (B) {

  // Estimate helpers are assigned from backend-bridge after load.
  B.maybeEstimate = B.maybeEstimate || null;
  B.fetchEstimate = B.fetchEstimate || null;
  B.invalidateEstimate = B.invalidateEstimate || null;

})(window.__UpscaleBridge = window.__UpscaleBridge || {});
