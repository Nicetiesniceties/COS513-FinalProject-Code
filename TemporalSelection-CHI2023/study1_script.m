Data = readtable('study1_data.csv');
Data = Data{:,:};
RemoveOutlier = true;

%% Data Pre-processing
pos = 0;
meantimeList = zeros(24, 1);
meanerrorList = zeros(24, 1);
ksResults = zeros(24, 1);
muList = zeros(24, 1);
sigmaList = zeros(24, 1);
para = zeros(24, 3); 
delNum = 0;
output_table = [];
for D_t = [0.4, 0.5, 0.6]
    for W_t = [0.1, 0.2]
        for R_t = [0.05, 0.1, 0.15, 0.2]
            pos = pos + 1;
            para(pos, 1) = D_t; para(pos, 2) = W_t; para(pos, 3) = R_t;
            tar = Data(:, 2)==D_t & Data(:, 3)==W_t & Data(:, 4)==R_t;
            timeList = Data(tar, 7);
            errorList = Data(tar, 8);
            meantimeList(pos) = mean(timeList);
            meanerrorList(pos) = mean(errorList);
            % Remove Outlier
            if (RemoveOutlier) 
                s = size(timeList, 1);
                stdtimeList = std(timeList);
                tar = timeList < meantimeList(pos) + 3 * stdtimeList & ...
                    timeList > meantimeList(pos) - 3 * stdtimeList;
                timeList = timeList(tar);
                errorList = errorList(tar);
                delNum = delNum + s - size(timeList, 1);
                meantimeList(pos) = mean(timeList);
                meanerrorList(pos) = mean(errorList);
            end
            % create new table
            output_table = [output_table; repmat(para(pos,:),size(timeList, 1), 1) timeList errorList];
            % Normalization and Normality Test
            listNorm_Time = (timeList-mean(timeList))/std(timeList);
            ksResults(pos) = kstest(listNorm_Time);
            % MLE
            phat = mle(timeList);
            muList(pos) = phat(1);
            sigmaList(pos) = phat(2);
        end
    end
end
meanerrorList = 1 - meanerrorList;
muList_abs = muList + para(:,3) - para(:,1);

%% Our Research
% Note D_t = para(:,1); W_t = para(:,2); R_t = para(:,3);
X = [para(:,1)-para(:,3), para(:,2), para(:,3)];
simple_mdl_mu_abs = fitlm(X,muList_abs);
simple_mdl_mu_abs
pred_mu_abs = predict(simple_mdl_mu_abs,[para(:,1)-para(:,3), para(:,2), para(:,3)]);

%corrcoef(muList, pred_mu)
%corrcoef(muList_abs, pred_mu_abs)
simple_mdl_mu_abs.ModelCriterion.AIC
simple_mdl_mu_abs.ModelCriterion.BIC
%https://au.mathworks.com/help/econ/information-criteria.html
sqrt(immse(muList_abs, pred_mu_abs))
mae(muList_abs, pred_mu_abs)

%% 3D plot related to para(:,1) and para(:,3)
Z_2D = zeros(3,4);
row = 1;
col = 1;
for D_t = [0.4, 0.5, 0.6]
    for R_t = [0.05, 0.1, 0.15, 0.2]
        tar = para(:, 1)==D_t & para(:, 3)==R_t;
        Z_2D(row,col) = mean(pred_mu_abs(tar));
        col = col + 1;
        if (col > 4) 
            col = 1;
            row = row + 1;
        end
    end
end
me = mesh([0.4, 0.5, 0.6], [0.05, 0.1, 0.15, 0.2], Z_2D.')
colormap spring
alpha 0.5
set(me,'LineWidth',1.5)
hold on;
sc = scatter3(para(:,1), para(:,3), muList+para(:,3)-para(:,1), 'o', 'k')
set(sc,'LineWidth',1.5)
set(gca,'FontSize',18)
hold off;

%% sigma fitting 
X = [para(:,1)-para(:,3), para(:,2), para(:,3)];
mdl_sigma = fitlm(X,sigmaList);
mdl_sigma

pred_sigma = predict(mdl_sigma,[para(:,1)-para(:,3), para(:,2), para(:,3)]);
mdl_sigma.ModelCriterion.AIC
mdl_sigma.ModelCriterion.BIC
sqrt(immse(sigmaList, pred_sigma))
mae(sigmaList, pred_sigma)

%% 2D plots
if (false)
    for i = (1:3)
    figure(i);
    plot(para(:,i), sigmaList, 'go');
    hold on;
    plot(para(:,i), pred_sigma, 'r*');
    hold off;
    end
end

% 3D plot related to para(:,1) and para(:,3)
Z_2D = zeros(3,4);
row = 1;
col = 1;
for D_t = [0.4, 0.5, 0.6]
    for R_t = [0.05, 0.1, 0.15, 0.2]
        tar = para(:, 1)==D_t & para(:, 3)==R_t;
        Z_2D(row,col) = mean(pred_sigma(tar));
        col = col + 1;
        if (col > 4) 
            col = 1;
            row = row + 1;
        end
    end
end
me = mesh([0.4, 0.5, 0.6], [0.05, 0.1, 0.15, 0.2], Z_2D.')
colormap winter
alpha 0.5
set(me,'LineWidth',1.5)
hold on;
sc = scatter3(para(:,1), para(:,3), sigmaList, 'o', 'k')
set(sc,'LineWidth',1.5)
set(gca,'FontSize',18)
hold off;

%% error fitting
fitMu_abs = predict(simple_mdl_mu_abs,[para(:,1)-para(:,3), para(:,2), para(:,3)]);
fitSigma = predict(mdl_sigma,[para(:,1)-para(:,3), para(:,2), para(:,3)]);

%%% No deviation version
%fun = @(t) exp(-(t-fitMu_abs).^2./(2.*fitSigma.^2))./(fitSigma.*sqrt(2.*pi));
%E = zeros(24, 1);
%for i = (1:24)
%    integ = integral(fun,0,para(i,2),'ArrayValued',true);
%    E(i) = 1 - integ(i);
%end
%mae(E, meanerrorList)
%corrcoef(E, meanerrorList)

%%% Deviation one level
E_erf = 1-(1/2).*(erf((para(:,2)-fitMu_abs)./(fitSigma.*sqrt(2))) + erf((fitMu_abs)./(fitSigma.*sqrt(2))));
R = corrcoef(E_erf, meanerrorList);
R.^2

%sqrt(immse(E_erf, meanerrorList))
mae(E_erf, meanerrorList)
%error_19 = sqrt(immse(E, meanerrorList));
%error_19


% Plot predicted vs. empirical
p1 = plot(linspace(0,1,50), linspace(0,1,50))
set(p1,'LineWidth',1.5)
hold on;
p2 = plot(meanerrorList, E_erf, 'k.', 'MarkerSize', 10)
set(gca,'FontSize',18)
hold off;

%% Error vs R_t
linecolors = parula(7)
i = 0;
for DD = [0.4, 0.5, 0.6]
    for WW = [0.1, 0.2]
        i = i + 1;
        select = para(:, 1)==DD & para(:, 2)==WW
        plot(para(select,3), meanerrorList(select), 'color', linecolors(i,:))
        hold on;
    end
end

%% 3D plot related to all parameters - mu
sc = scatter3(para(:,1), para(:,3), muList_abs, '*', 'r')
set(sc,'LineWidth',1.5)
hold on;
spre = scatter3(para(:,1), para(:,3), fitMu_abs, 'o', 'k')
hold off;

%% 3D plot related to all parameters - sigma
sc = scatter3(para(:,1), para(:,3), sigmaList, '*', 'r')
set(sc,'LineWidth',1.5)
hold on;
spre = scatter3(para(:,1), para(:,3), fitSigma, 'o', 'k')
hold off;

%% Simplified Model
% Note D_t = para(:,1); W_t = para(:,2); R_t = para(:,3);
X = [para(:,1), para(:,3)];
simple_mdl_mu_abs = fitlm(X,muList_abs);
simple_mdl_mu_abs
pred_mu_abs = predict(simple_mdl_mu_abs,[para(:,1), para(:,3)]);

simple_mdl_mu_abs.ModelCriterion.AIC
simple_mdl_mu_abs.ModelCriterion.BIC
mae(muList_abs, pred_mu_abs)

% sigma fitting -- Note: x(1) = c_sigma;
X = [para(:,1)];
simple_mdl_sigma = fitlm(X,sigmaList);
simple_mdl_sigma

pred_sigma = predict(simple_mdl_sigma,X);
simple_mdl_sigma.ModelCriterion.AIC
simple_mdl_sigma.ModelCriterion.BIC
mae(sigmaList, pred_sigma)

fitMu_simple_abs = predict(simple_mdl_mu_abs,[para(:,1), para(:,3)]);
fitSigma_simple = predict(simple_mdl_sigma,[para(:,1)]);
E_erf16 = 1-(1/2).*(erf((para(:,2)-fitMu_simple_abs)./(fitSigma_simple.*sqrt(2))) ...
    + erf((fitMu_simple_abs)./(fitSigma_simple.*sqrt(2))));
R = corrcoef(E_erf16, meanerrorList);
R.^2
%error_16 = sqrt(immse(E_erf16, meanerrorList))
mae(E_erf16, meanerrorList)

p1 = plot(linspace(0,1,50), linspace(0,1,50))
set(p1,'LineWidth',1.5)
hold on;
plot(meanerrorList, E_erf16, 'k.', 'MarkerSize', 10)
hold off;

%% Interact Model
X = [para(:,1), para(:,3), para(:,1).*para(:,3)];
simple_mdl_mu_abs = fitlm(X,muList_abs);
simple_mdl_mu_abs
pred_mu_abs = predict(simple_mdl_mu_abs,[para(:,1), para(:,3), para(:,1).*para(:,3)]);

simple_mdl_mu_abs.ModelCriterion.AIC
simple_mdl_mu_abs.ModelCriterion.BIC
mae(muList_abs, pred_mu_abs)

% sigma fitting -- Note: x(1) = c_sigma;
X = [para(:,1)];
simple_mdl_sigma = fitlm(X,sigmaList);
simple_mdl_sigma

pred_sigma = predict(simple_mdl_sigma,X);
simple_mdl_sigma.ModelCriterion.AIC
simple_mdl_sigma.ModelCriterion.BIC
mae(sigmaList, pred_sigma)

fitMu_simple_abs = predict(simple_mdl_mu_abs,[para(:,1), para(:,3), para(:,1).*para(:,3)]);
fitSigma_simple = predict(simple_mdl_sigma,[para(:,1)]);
E_erf16 = 1-(1/2).*(erf((para(:,2)-fitMu_simple_abs)./(fitSigma_simple.*sqrt(2))) ...
    + erf((fitMu_simple_abs)./(fitSigma_simple.*sqrt(2))));
R = corrcoef(E_erf16, meanerrorList);
R.^2
%error_16 = sqrt(immse(E_erf16, meanerrorList))
mae(E_erf16, meanerrorList)

p1 = plot(linspace(0,1,50), linspace(0,1,50))
set(p1,'LineWidth',1.5)
hold on;
plot(meanerrorList, E_erf16, 'k.', 'MarkerSize', 10)
set(gca,'FontSize',18)
hold off;

%% Simulated Effect of Temporal Factors - One Model (INTERACT MODEL)
X = [para(:,1), para(:,3), para(:,1).*para(:,3)];
simple_mdl_mu_abs = fitlm(X,muList_abs);

% sigma fitting -- Note: x(1) = c_sigma;
X = [para(:,1)];
mdl_sigma = fitlm(X,sigmaList);

linecolors = cool(21);
i = 0;
for DD = [0.4:0.01:0.6]
    i = i + 1;
    WW = 0.1; % parameter here {0.1, 0.2}
    D_sim = repmat(DD, 21, 1);
    W_sim = repmat(WW, 21, 1);
    R_sim = [0:0.01:0.2]';
    para_sim_mu = [D_sim, R_sim, D_sim .* R_sim];
    para_sim_sigma = [D_sim];
    fitMu_sim = predict(simple_mdl_mu_abs,para_sim_mu);
    fitSigma_sim = predict(mdl_sigma,para_sim_sigma);
    E_erf_sim = 1-(1/2).*(erf((W_sim-fitMu_sim)./(fitSigma_sim.*sqrt(2))) + erf((fitMu_sim)./(fitSigma_sim.*sqrt(2))));
    p_sim = plot(R_sim, E_erf_sim,'color', linecolors(i,:),'LineWidth',1);
    ylim([0 1])
    hold on;
end
colormap(linecolors)
colorbar('Ticks',[0:0.25:1], 'TickLabels',[0.4:0.05:0.6])
set(gca,'FontSize',18)


