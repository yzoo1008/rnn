%%
clear;
clc;
close all;
cd 'C:\Users\KYJ\OneDrive - SNU\대학원\SK_Rod'
%% Data Load
load('./data/Day', 'Day');
load('./data/Oil_Daily', 'Oil_d');
load('./data/Gas_Daily', 'Gas_d');
load('./data/Water_Daily', 'Water_d');
load('./data/PF_Daily', 'PF_d');
load('./data/RS_Daily', 'RS_d');
%% Plot
h = figure;
hold on; grid on;
plot(Day, Oil_d, 'o', 'Color', [0.00,0.45,0.74]);
plot(Day, Gas_d, 'o', 'Color', [0.85,0.33,0.10]);
plot(Day, Water_d, 'o', 'Color', [0.93,0.69,0.13]);
plot(Day, PF_d, 'o', 'Color', [0.49,0.18,0.56]);
plot(Day, RS_d, 'o', 'Color', [0.47,0.67,0.19]);
legend({'Oil', 'Gas', 'Water', 'Pump Fillage', 'Rod Speed'}, 'Location', 'Northeast', 'Fontsize', 10);
set(h, 'Position',[100 100 1200 600]);

% export_fig('./figure/avg data', '-transparent','-png','-r300'); 
%% Histogram
figure(); set(gcf,'units','normalized','outerposition',[0 0 1 1]);

subplot(2, 3, 1);
histogram(Oil_d);
xlabel('Oil Rate (BBL/d)', 'Fontsize', 14);

subplot(2, 3, 2);
histogram(Gas_d);
xlabel('Gas Rate (MCF/d)', 'Fontsize', 14);

subplot(2, 3, 3);
histogram(Water_d);
xlabel('Water Rate (BBL/d)', 'Fontsize', 14);

subplot(2, 3, 4);
histogram(PF_d);
xlabel('Pump Fillage (%)', 'Fontsize', 14);

subplot(2, 3, 5);
histogram(RS_d);
xlabel('Rod Speed', 'Fontsize', 14);

% export_fig('./figure/(Raw)Histogram', '-transparent','-png','-r300');
%% Data 병합
data = zeros(5, 239);
for i = 1:239
    data(1, i) = Oil_d(i, 1);
    data(2, i) = Gas_d(i, 1);
    data(3, i) = Water_d(i, 1);
    data(4, i) = PF_d(i, 1);
    data(5, i) = RS_d(i, 1);
end
%% 훈련 데이터와 테스트 데이터 나누기, 9:1 비율
numTimeStepsTrain = floor(0.9*numel(data)/5);

dataTrain = data(:, 1:numTimeStepsTrain+1);
dataTest = data(:, numTimeStepsTrain+1:end);
dataKnown = data(:, numTimeStepsTrain+1);
%% Normalization, Standardization, MinMax-scaling
% mu = mean(dataTrain, 1);
% sig = std(dataTrain, 0, 1);
% dataTrainStandardized = (dataTrain - mu) ./ sig;

mu = mean(dataTrain, 2);
sig = std(dataTrain, 0, 2);
dataTrainStandardized = (dataTrain - mu) ./ sig;

% min_v = min(dataTrain, [], 2);
% max_v = max(dataTrain, [], 2);
% dataTrainStandardized = (dataTrain - min_v) ./ (max_v - min_v);

%% Plot
figure(); set(gcf,'units','normalized','outerposition',[0 0 1 1]);

subplot(2, 3, 1);
histogram(dataTrainStandardized(1, :));
xlabel('Oil Rate (BBL/d)', 'Fontsize', 14);

subplot(2, 3, 2);
histogram(dataTrainStandardized(2, :));
xlabel('Gas Rate (MCF/d)', 'Fontsize', 14);

subplot(2, 3, 3);
histogram(dataTrainStandardized(3, :));
xlabel('Water Rate (BBL/d)', 'Fontsize', 14);

subplot(2, 3, 4);
histogram(dataTrainStandardized(4, :));
xlabel('Pump Fillage (%)', 'Fontsize', 14);

subplot(2, 3, 5);
histogram(dataTrainStandardized(5, :));
xlabel('Rod Speed', 'Fontsize', 14);

% export_fig('./figure/(Raw)Histogram-Normalization', '-transparent','-png','-r300'); 
%% 예측 변수와 응답변수 준비
XTrain = dataTrainStandardized(:, 1:end-1);
YTrain = dataTrainStandardized(4, 2:end);
%% LSTM 네트워크 정의
numFeatures = 5;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod', 125, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% 네트워크 훈련
net = trainNetwork(XTrain,YTrain,layers,options);
%% 미래의 시간 스텝 예측
% mu = mean(dataTest, 1);
% sig = std(dataTest, 0, 1);
% dataTestStandardized = (dataTest - mu) ./ sig;
dataTestStandardized = (dataTest - mu) ./ sig;
% dataTestStandardized = (dataTest - min_v) ./ (max_v - min_v);

XTest = dataTestStandardized(:, 1:end-1);

% predictAndUpdateState는 현재 스텝의 예측값을 다음 예측 시, 입력값으로 사용
net = predictAndUpdateState(net, XTrain);
[net, Y_hat] = predictAndUpdateState(net, dataKnown);
XTest(4, 1) = Y_hat(1, 1);

numTimeStepsTest = numel(XTest)/5;
for i = 1:numTimeStepsTest
    [net, YPred(1, i)] = predictAndUpdateState(net, XTest(:, i),'ExecutionEnvironment','cpu');
    XTest(4, i+1) = YPred(1, i);
end
% YPred = sig(1, 2:end).*YPred + mu(1, 2:end);
YPred = sig(4)*YPred + mu(4);
% YPred = (max_v(4) - min_v(4)) * YPred + min_v(4);
%%
YTest = dataTest(4, 2:end);
rmse = sqrt(mean((YPred-YTest).^2))
%% Plot
n_train = size(dataTrain, 2);
n_test = size(YTrain, 2);

figure
plot(Day(1 : n_train), dataTrain(4, :));
hold on
plot(Day(n_train+1 : end), YPred,'.-');
hold off
xlabel("Time")
ylabel("Pump Fillage")
title("Forecast")
legend(["Observed" "Forecast"])
%% 예측된 값과 테스터 데이터 비교
figure(); set(gcf,'units','normalized','outerposition',[0 0 1 1]);
subplot(2,1,1)
plot(Day(n_train+1 : end), YTest)
hold on
plot(Day(n_train+1 : end), YPred,'.-')
hold off
legend(["Observed" "Forecast"], "Fontsize", 12)
ylabel("Pump Fillage (%)", "Fontsize", 14)
xlabel("Date", "Fontsize", 14)
title("Forecast")

subplot(2,1,2)
stem(Day(n_train+1 : end), YPred - YTest)
xlabel("Date", "Fontsize", 14)
ylabel("Error", "Fontsize", 14)
title("RMSE = " + rmse)
export_fig('./figure/LSTM-normalization', '-transparent','-png','-r300'); 