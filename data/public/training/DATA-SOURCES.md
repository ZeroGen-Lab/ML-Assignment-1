# Course public training data sources

Version: course-uci-20261014-v3
Dataset ID: 0f23098f6a76cf60424ca04bc1bfe1d7e96ad1ef1a59a632dd2b319ab162c671

Public continuous CSVs contain 200 representative formal-train rows each.
LR and SVM share the same Spambase split, features, labels and public 200 rows.
Server fit uses the complete formal training split; public 200 is not the whole formal fit.
Features are already standardized using formal train only; do not scale them again.
KMeans Rice class labels and all formal holdout rows/truth are not course-published.
HMM publishes 10 intact sequences of 20 (200 rows); formal train/test remains 60/30.
HMM states 0..2 and observations 0..3 are supervised; true parameters are not published.
Original UCI labels are public; withholding is not a secrecy or anti-cheating guarantee.

## Spambase
Authors: Hopkins, M.; Reeber, E.; Forman, G.; Suermondt, J.
DOI: https://doi.org/10.24432/C53G6X
Source: https://archive.ics.uci.edu/dataset/94/spambase
License: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
Downloaded: 2026-10-07

## Combined Cycle Power Plant
Authors: Tufekci, P.; Kaya, H.
DOI: https://doi.org/10.24432/C5002N
Source: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
License: CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
Downloaded: 2026-10-07

## Rice Cammeo and Osmancik
Source: https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
Official archive: https://archive.ics.uci.edu/static/public/545/rice%2Bcammeo%2Band%2Bosmancik.zip
Archive SHA256: fe94e42046b829de21b92b0ffb6a22774fde021328cae799faa802a14e8dbed9
Downloaded: 2026-10-14 UTC
License: Creative Commons Attribution 4.0 International — https://creativecommons.org/licenses/by/4.0/
Citation: Cinar and Koklu (2019), DOI 10.18201/ijisae.2019355381; UCI DOI 10.24432/C5MW4Z
Features f0..f6: Area, Perimeter, Major_Axis_Length, Minor_Axis_Length, Eccentricity, Convex_Area, Extent
Changes: conflict/exact-dedup cleaning, group-safe fixed split and train-only z-score.
