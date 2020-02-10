FILE=$1

if [ $FILE == "afhq" ]; then
    URL=https://www.dropbox.com/s/1j5pi6l7x1sb7f2/afhq.zip?dl=0
    ZIP_FILE=./data/afhq/afhq.zip
    mkdir -p ./data/afhq/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/afhq/
    rm $ZIP_FILE
else
    echo "Available argument is afhq...!"
    exit 1
fi