cd Data
# DALIA
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip -O dalia.zip
unzip dalia.zip
rm dalia.zip
mv PPG_FieldStudy DaLia
# WESAD
wget https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download -O wesad.zip
unzip wesad.zip
rm wesad.zip
# BAMI
wget https://github.com/HeewonChung92/CNN_LSTM_HeartRateEstimation/archive/refs/heads/master.zip -O bami.zip
unzip bami.zip
rm bami.zip
mv CNN_LSTM_HeartRateEstimation-master BAMI