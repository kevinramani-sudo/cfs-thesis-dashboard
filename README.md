# cfs-thesis-dashboard
Interactive dashboard and code archive for CFS framing cost-carbon-time optimisation thesis.
# CFS Framing Optimisation Dashboard

This repository contains the code, exported model outputs, figures, and browser-based dashboard developed for a Master of Engineering thesis on cold-formed steel residential framing optimisation.

The dashboard presents CFS framing process cost, CFS framing A1–A5 embodied carbon, and CFS framing process-hours for a bounded residential case study. It uses exported CSV and JSON outputs from the implemented NSGA-II optimisation framework.

## Public Dashboard

Dashboard link:

https://kevinramani-sudo.github.io/cfs-thesis-dashboard/dashboard/

## Repository Contents

- `index.html` — browser-based dashboard interface
- `dashboard_config.json` — dashboard configuration and model metadata
- `data/` — exported optimisation, Pareto-front, sensitivity, robustness, and ranking outputs
- `figures/` — dashboard and thesis figures
- `scripts/` — Python model script used to generate the optimisation outputs

## Scope Note

The dashboard does not rerun the NSGA-II optimisation. It visualises and interrogates exported model outputs from the reported thesis run. The dashboard is intended as a decision-support and interpretation interface, not as a commercial costing tool, structural design checker, or full-building life-cycle assessment tool.

## Thesis Boundary

- Cost boundary: CFS framing process cost
- Carbon boundary: CFS framing A1–A5 embodied carbon
- Time boundary: CFS framing process-hours
- Case-study boundary: fixed single-storey residential CFS framing case
- Reference scenario: S2 model-calculated benchmark

## Author

Kevin Ramani  
Master of Engineering Civil  
University of Waikato
