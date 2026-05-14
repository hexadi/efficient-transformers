# Gemma4 Export Status

Gemma4 export support has been validated for `google/gemma-4-E2B-it`.

## Validated

- Export passed with:

```bash
python scripts/smoke_gemma4_export.py --model google/gemma-4-E2B-it
```

- `onnx.checker.check_model(path)` passed. Path-based checking was used because the exported ONNX model is larger than 2 GiB.
- PyTorch/QEff dummy outputs passed shape validation:
  - logits: `[1, 1, 262144]`
  - retained vision state: `[1, 252, 768]`
  - image index output: `[1, 1]`

## Known Limit

ONNX Runtime CPU cannot load the exported QEfficient custom-op graph. The validated graph contains QEfficient custom ops such as `CustomRMSNorm`, and ORT CPU fails during custom-op shape/type inference.

## Next Step

Run QEfficient compile/runtime validation on target hardware before claiming full Gemma4 runtime support.
