cwd=$(pwd)
cd `dirname $0`/..

GDRIVE_URL='https://drive.google.com/drive/folders/1hzTCaXh6qtZgoOC6_VPVZOBsA_fKcBft?usp=drive_link'
python -m gdown --folder "$GDRIVE_URL"

rsync -avr ./tmp_prt_lm/ ./prt_lm
rm -r ./tmp_prt_lm
cd $cwd