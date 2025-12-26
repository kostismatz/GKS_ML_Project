# ffmpeg -i test.wav -ar 44100 test1.wav

if [ ! -d ./fixed ]
then
    echo "Folder structure not found! creating..."
    mkdir -p ./fixed/fold1
    mkdir -p ./fixed/fold2
    mkdir -p ./fixed/fold3
    mkdir -p ./fixed/fold4
    mkdir -p ./fixed/fold5
    mkdir -p ./fixed/fold6
    mkdir -p ./fixed/fold7
    mkdir -p ./fixed/fold8
    mkdir -p ./fixed/fold9
    mkdir -p ./fixed/fold10
    echo "done"
fi

for i in fold*/*.wav; 
    # do ffmpeg -i "$i" -c:v copy -c:a copy -ar 44100 "fixed/$i";
    do ffmpeg -i "$i" -ar 16000 "fixed/$i";
done