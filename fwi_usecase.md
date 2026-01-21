# Extremes DT FWI use case

The goal is to write a processing service (Delta Twin or Insula Intellect) that processes the weather-induced extremes Digital Twin data forecast to create a Fire Weather Index (FWI) forecast

Data inputs:
- Extremes DT

Services:
- HDA or Polytope
- Insula Intellect or Delta Twin

Outputs:
- Regular fire weather index plots over some AOI
- Published insula processing service or delta twin component

### Note: Polytope is under maintenance

Polytope is under maintenance until 28 Jan, so use HDA for data access.

### Note: No logs in Insula Intellect

Insula intellect does not yet support logs for processing services, which makes debugging impossible.

### Note: Extremes DT is currently unavailable

Not sure when it will be back.

## Building a Delta Twin component using HDA

### Requirements

HDA authentication is supported in destinepyauth > 1.1.0
```
destinepyauth
```

### Data

Find the collection ID via the UI
```
HDA_STAC_ENDPOINT="https://hda.data.destination-earth.eu/stac/v2"
COLLECTION_ID = "EO.ECMWF.DAT.DT_EXTREMES"
```
However, extremes DT is currently down so using the Fire Risk Map (MSG)
```
COLLECTION_ID = "EO.EUM.DAT.MSG.LSA-FRM"
```
Variables

### AOI
Cobières Massif is a region in France that [recently suffered from wild fire](https://en.wikipedia.org/wiki/2025_Corbi%C3%A8res_Massif_wildfire).
```
bbox = [2.10, 42.65, 3.25, 43.35]
```
not relevant here because the data is with 'phony' dimensions

## Creating a Delta Twin component locally

### inputs.json

All we need as inputs are DESP auth credentials

```
{
  "user": {
    "type": "string",
    "value": "johnsmith"
  },
  "password": {
    "type": "string",
    "value": "XXXXXX"
  }
}
```
### manifest.json

guide [here](https://deltatwin.destine.eu/docs/tutorials/basic/tutorial_basic_step3)

- Fill in ownership details and name at the top
- No resources - these are more for connecting inputs (not explained well in docs)
- 