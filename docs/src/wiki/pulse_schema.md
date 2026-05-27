# Typed Pulse Schema

The compiler now expects a typed pulse schema centered on logical gate recipes.

## Top-Level Layout

```yaml
schema_version: "1.0"
pulse:
  defaults:
    schedule_policy: parallel
    xy_carrier_freq_Hz: 5.0e9
    ro_carrier_freq_Hz: 6.5e9

  gates:
    sx:
      recipe_type: sx
      duration_ns: 28.0
      amplitude_Hz: 10.5e6

    virtual_z:
      recipe_type: virtual_z

    cz:
      recipe_type: cz
      duration_ns: 52.0
      amplitude_Hz: 20.0e6

  channel_overrides:
    XY_0:
      sx:
        amplitude_Hz: 10.8e6
```

## Semantics

- `defaults`
  - global fallback values that are not gate-specific calibrations
- `gates`
  - canonical typed recipe definitions per logical gate
- `channel_overrides`
  - patch layer applied after the physical channel is known

Resolution order is:

1. logical gate recipe from `gates`
2. channel-specific patch from `channel_overrides`
3. defaults for non-gate-specific fallback values only

## Required Recipe Contracts

## `SX`

- requires `duration_ns`
- requires `amplitude_Hz`
- may include `shape`, `sigma_fraction`, `drag_beta`, `carrier_freq_Hz`,
  `phase_rad`

## `CZ`

- requires `duration_ns`
- requires `amplitude_Hz`
- may include `shape`, `edge_ns`, `target_conditional_phase_rad`

## `VirtualZ`

- uses `recipe_type: virtual_z`
- must not include `duration_ns`
- must not include `amplitude_Hz`
- lowers to frame/phase updates only

## Rejected Legacy Patterns

The following user-facing pulse styles are rejected:

- `channels` / `carriers` / `waveforms` / `operations`
- top-level flat pulse knobs instead of `defaults`
- any `amp_scale`-style recipe field

Use explicit `amplitude_Hz` instead.
