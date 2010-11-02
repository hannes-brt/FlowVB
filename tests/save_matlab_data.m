function save_matlab_data( filename, matlab_variables,...
    output_names)
%SAVE_MATLAB_DATA Save data from a matlab session to disk

output_struct=struct;

for i=1:length(matlab_variables)
    s = size(matlab_variables{i});
    if length(s)==3
        temp = NaN(s([3 1 2]));
        for k=1:s(3)
            temp(k,:,:) = matlab_variables{i}(:,:,k);
        end
        matlab_variables{i} = temp;
    end
    output_struct.(output_names{i}) = matlab_variables{i};
end

save(filename, '-struct', 'output_struct'); 
end

