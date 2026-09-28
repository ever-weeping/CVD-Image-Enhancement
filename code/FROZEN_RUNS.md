# Frozen run provenance

The paper's statistics are reproducible from the three ignored run folders
below. Training checkpoints were intentionally not saved to conserve local
storage, so the claim is **analysis reproducibility from frozen per-image
outputs**, not bitwise training reproducibility on MPS.

| Run | Purpose | Git commit at start | Device |
|---|---|---|---|
| `runs/pilot_v2_20260720_fixed` | primary comparison, branch/loss ablations and stress set | `cfff09461459138a8e37472ae2be95ed74559f48` | MPS |
| `runs/condition_probe_20260720` | inference-only FiLM axis/control probes | `5bbb24cd125970e53bcb74ddcc03a9e5ad37b4f1` | MPS |
| `runs/no_mask_ablation_20260720` | full mask-chain removal ablation | `7c789924a4b51d51245dd2f9defab678a4c3d4bd` | MPS |

All three runs use the same `split_manifest.csv` (SHA-256
`88a472c51090bc33f180333f6c87f728583be78c09ff53ddf271c5f8a8819110`).

## SHA-256 manifest

```text
e293150753fad4167123c52e6b09e6ed9247edc66bd1560d467104eb2e372b1d  runs/pilot_v2_20260720_fixed/config.json
5922e96389dbc8b303a41dddbcf23b9d75344cf9e68fc50464720b50bd924b19  runs/pilot_v2_20260720_fixed/per_image_metrics.csv
6bd68f929b3684c29f6ce6fbe9988115938db1a39acd203948f3e56e1e775b3e  runs/pilot_v2_20260720_fixed/summary_overall.csv
35c60f43e990e20fe6a85d768e65f80682e7ceda902fc92541a437cc045decb3  runs/pilot_v2_20260720_fixed/paired_statistics.csv
7d6e82593115ce5eb1d7176461198c19629a785082e77450890c8357b19cfe73  runs/condition_probe_20260720/config.json
8eba4fbbaebfbe562fb439a589ee69b5be485c281a3f9d82f82ffac4ee95b31a  runs/condition_probe_20260720/per_image_metrics.csv
bf73ffaf2fa8746dddcbec62237f1eefc5ea4e86fcff03e4020a7dfb02e941c1  runs/condition_probe_20260720/summary_overall.csv
7b15e36acb64a61b2c62b94204474f1d0be45ca4a580814194e6a2e3372f0026  runs/condition_probe_20260720/paired_statistics.csv
66b357963e2b4ac6e4aa14916e823df0ea0fb526fd8ed5ce353e34c2295b22a6  runs/no_mask_ablation_20260720/config.json
b13fac33a2f216f8be41562a556a42783327069b5210e80d088ec8474db60deb  runs/no_mask_ablation_20260720/per_image_metrics.csv
c74dcae3feeb50925917ab0df4d6e38bf22dd0f6984605c9074bc9f86b46ac6b  runs/no_mask_ablation_20260720/summary_overall.csv
```

The qualitative figure uses the deterministic `_00` sample saved for the
protan and deutan endpoints; it is not selected by metric performance.
