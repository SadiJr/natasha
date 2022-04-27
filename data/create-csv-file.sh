#!/usr/bin/env bash

set -e

unzip 'Requirements data sets (user stories).zip'

for i in *.txt; do
  iconv -f WINDOWS-1252 -t UTF-8 "$i" > "$i".conv
done

rm *.txt
cat *.conv > concat.txt
rm *.conv

dos2unix concat.txt

sed -i '/^$/d' concat.txt
sed -i -r "s/\\\"/\\'/g" concat.txt
sed -i 's/^/"/' concat.txt
sed -i 's/$/",/' concat.txt
