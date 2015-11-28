function data_view3(data, start_row, targets)
    grid on; hold on;
    data_size = size(data,2);
    j = start_row - 1;
    for i=1:data_size
        plot3(data(j+1,i),data(j+2,i),data(j+3,i),line_spec(targets, i));
    end
    view(3)
end

function ls = line_spec(targets, column)
    if targets(1,column) == 1
        ls = 'b*';
    elseif targets(2,column) == 1
        ls = 'co';
    else
        ls = 'mx';
    end    
end    
