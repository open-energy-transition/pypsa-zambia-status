# Zambia hydropower reference data

The reference data are published by the Energy Regulation Board (ERB)
of Zambia and are used for historical generation validation.

## Sources

### Annual generation

[Energy Sector Report 2023](https://www.erb.org.zm/wp-content/uploads/files/esr2023.pdf)

- Table 3-3, p. 13: generation from major hydropower stations.
- Table 3-6, p. 15: generation from independent power producers,
  including Itezhi Tezhi.

Annual generation is reported in GWh.

### Monthly generation

[Mid-Year Statistical Bulletin 2023](https://www.erb.org.zm/wp-content/uploads/midYearStatBullet2023.pdf)

- Table 14, p. 21: Kafue Gorge Upper, Kariba North Bank and Victoria Falls.
- Table 17, pp. 22-23: Itezhi Tezhi, Kafue Gorge Lower and Kariba North Bank Extension.

Monthly observations cover January to June 2023.
Reported MWh values are converted to GWh.

### Installed capacity

Mid-Year Statistical Bulletin 2023, Appendix 2 (PDF page 29).

| Reference station | Installed capacity (MW) |
| --- | ---: |
| Itezhi Tezhi | 120 |
| Kafue Gorge Upper | 990 |
| Kafue Gorge Lower | 750 |
| Kariba North Bank and Extension | 1080 |
| Victoria Falls | 108 |

The capacities in ERB_station_metadata_2023.csv were verified against
this independent reference.

## Methodology and limitations

- Kariba North Bank and its extension are aggregated in both generation
  and installed capacity (720 + 360 MW).
- Kafue Gorge Upper and Lower are compared separately whenever the
  model representation allows it.
- Capacity factors are derived from generation and nominal installed
  capacity, rather than independently reported by ERB.
- Reference capacity factors use the nominal capacity reported at
  mid-year throughout 2023, without adjusting for commissioning dates.
  This particularly affects Kafue Gorge Lower.
- Reference generation is used for validation, not as a dispatch constraint.
