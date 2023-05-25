%At first, we would like to figure out the relationship between gender and
%survival by Naive Bayes.
%Initialize label and decision
train = readtable('train.csv');
gender = table2array(rows2vars(train(:,"Sex")));
gender = gender(1,2:end);
labelGender = [zeros(1,length(find(strcmp('female',gender)))),...
    ones(1,length(find(strcmp('male',gender))))];
survival = rows2vars(train(:,"Survived"));
survival = table2array(survival(1,2:end));
decisionSurvival = [survival(find(strcmp('female',gender))),survival(find(strcmp('male',gender)))];

%Classifier: Calculate the p(X=0|Y=1),p(X=1|Y=1)
pFeSur = sum(labelGender==0 & decisionSurvival==1)/length(find(decisionSurvival==1));
pMSUR = sum(labelGender==1 & decisionSurvival==1)/length(find(decisionSurvival==1));
p = [pFeSur,pMSUR]；
[maxp, ind] = max(p);

%Load Test data
test = readtable('test.csv');
sex = table2array(rows2vars(test(:,"Sex")));
sex = sex(1,2:end);
labelSex = [zeros(1,length(find(strcmp('female',sex)))),...
    ones(1,length(find(strcmp('male',sex))))];
s = [0,1];
y = [length(find(strcmp('female',sex))),length(find(strcmp('male',sex)))];
if ind == 1
    numFeSur = s(2)*length(find(strcmp('female',sex)));
    numMSur = s(1)*length(find(strcmp('male',sex)));
else
    numFeSur = s(1)*length(find(strcmp('female',sex)));
    numMSur = s(2)*length(find(strcmp('male',sex)));
end
num = [numFeSur,numMSur];

%Plot the result
figure(1)
bar(s,num);hold on;
bar(s,y-num);hold off;
xlabel('not survived=0 survived=1');
ylabel('numbers');
legend('female survived','male not survived');
saveas(gcf,'gender survival.png');

%From the estimation of Naive Bayes, we can give a simple predicition that
%all the females will be survived and all the males will not be survived.
%This is absolutely wrong. To change this situation, instead of applying
%only sex as a feature, others like age and Pclass are also need to be
%considered. Before we start merging them, we would like to se how these
%features work individually.

