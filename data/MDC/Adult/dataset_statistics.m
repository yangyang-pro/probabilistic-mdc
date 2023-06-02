clear;clc;close all;fclose('all');
load('Adult.mat');
[m,q] = size(target);
class_per_dim = cell(q,1);%class labels in each dimension
num_per_dim = zeros(q,1);%number of class labels in each dimension
for dd=1:q
    temp = target(:,dd);
    class_per_dim{dd} = unique(temp);
    num_per_dim(dd) = length(class_per_dim{dd});
end
disp('General information:')
disp(['Data Set    : ', data_name]);
disp(['#Examples   : ', num2str(m)]);
disp(['#Dimensions : ', num2str(q)]);
tmpC = unique(num_per_dim);
if length(tmpC)==1
    disp(['#Labels/Dim.: ', num2str(num_per_dim(1))]);
else
    tmp_str = '#Labels/Dim.: ';
    for dd=1:q
        tmp_str = [tmp_str, num2str(num_per_dim(dd)), ','];
    end
    disp(tmp_str);
end
tmp_str = '#Features   : ';
if ~isempty(data_type.c)
    tmp_str = [tmp_str, num2str(length(data_type.c)),'n,'];
end
if ~isempty(data_type.b)
    tmp_str = [tmp_str, num2str(length(data_type.b)),'b,'];
end
if ~isempty(data_type.d_wo_o)
    tmp_str = [tmp_str, num2str(length(data_type.d_wo_o)),'x,'];
end
if ~isempty(data_type.d_w_o)
    tmp_str = [tmp_str, num2str(length(data_type.d_w_o)),'xo,'];
end
disp(tmp_str);

disp('NOTE1: If all class spaces have the same number of class labels, then only this number is recorded;');
disp('       Otherwise, the number of class labels in each class space is recorded in turn.')
disp('NOTE2: n, b, x and xo denote numeric, binary, nominal and nominal with ordinal relationship features, respectively.')
disp('          ');

disp('Detailed information:')
for dd=1:q
    temp = target(:,dd);
    tmp_str = ['Dim.',num2str(dd),': '];
    for aa=1:num_per_dim(dd)
        tmp_str = [tmp_str,num2str(class_per_dim{dd}(aa)),'/'];
    end
    disp(tmp_str);
    tmp_str = '';
	for aa=1:num_per_dim(dd)
        tmp_str = [tmp_str, 'Label(',num2str(class_per_dim{dd}(aa)),'): ',num2str(sum(temp==class_per_dim{dd}(aa))),', '];
    end
    disp(tmp_str);
end
disp('          ');
for dd=data_type.d_wo_o
    temp = data.orig(:,dd);
    class_set = unique(temp);
    num_value = length(class_set);
    tmp_str = ['Attr.',num2str(dd),': '];
    for aa=1:num_value
        tmp_str = [tmp_str,num2str(class_set(aa)),'/'];
    end
    disp(tmp_str);
    tmp_str = '';
	for aa=1:num_value
        tmp_str = [tmp_str, 'Value(',num2str(class_set(aa)),'): ',num2str(sum(temp==class_set(aa))),', '];
    end
    disp(tmp_str);
end