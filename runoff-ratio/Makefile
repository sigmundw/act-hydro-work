summer:
	time parallel -j 14 python runoff_ratio_mjja.py {} ::: `seq 13`

winter:
	time parallel -j 14 python runoff_ratio_djfm.py {} ::: `seq 13`

all:
	time parallel -j 14 python runoff_ratio_mjja.py {} ::: `seq 13`
	time parallel -j 14 python runoff_ratio_djfm.py {} ::: `seq 13`
