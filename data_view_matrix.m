function data_view_matrix(data, targets)
    column_size = size(data, 2);
    group = zeros(1, column_size);
    for i = 1:column_size
        group(1, i) = group_no(targets, i);
    end    
    gplotmatrix(transpose(data), [], transpose(group), ['c','b','m'], [], [], false); 
    var_names = {
        'Alc.'; 'M.Acid'; 'Ash'; 'Alc.Ash'; 'Mag.'; 
        'T.Phen.'; 'Flav.'; 'N.Phen.'; 
        'Proant.'; 'Color'; 'Hue'; 'OD280'; 'Proline'
    };
    var_size = size(var_names, 1);
    text(((1:1:var_size)/13)-0.07, repmat(-.05,1,var_size), var_names, 'FontSize',8);
    text(repmat(-.04,1,var_size), ((var_size:-1:1)/13)-0.06, var_names, 'FontSize',8, 'Rotation',90);
end

function result = group_no(targets, column)
    if targets(1,column) == 1
        result = 1;
    elseif targets(2,column) == 1
        result = 2;
    else
        result = 3;
    end    
end    


