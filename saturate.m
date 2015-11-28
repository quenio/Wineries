function s = saturate(output)
    row_size = size(output, 1);
    col_size = size(output, 2);
    s = zeros(row_size, col_size);
    for i = 1:col_size
        v = output(:, i);
        if v(1) > v(2) && v(1) > v(3)
            s(1, i) = 1;
        elseif v(2) > v(1) && v(2) > v(3)
            s(2, i) = 1;
        else
            s(3, i) = 1;
        end    
    end
end

    