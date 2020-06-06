function run_toy()
    fprintf('running ZooBP\n');
    L = csvread('adjlist.csv');
    
    user_priors = csvread('userPriors.csv');
    prod_priors = csvread('prodPriors.csv');
    
    [~,idx] = sort(user_priors(:,1));
    sorted_user_priors = user_priors(idx,:);
    %disp(sorted_user_priors);
    
    [~,idx] = sort(prod_priors(:,1));
    sorted_prod_priors = prod_priors(idx,:);  
    
    csvwrite('sortedUserPriors.csv',sorted_user_priors); 
    csvwrite('sortedProductPriors.csv',sorted_prod_priors);
    
    %sorted_user_priors(:,1) = sorted_user_priors(:,1) - 0.5 * ones(size(sorted_user_priors, 1), 1);
    %sorted_user_priors(:,2) = sorted_user_priors(:,2) - 0.5 * ones(size(sorted_user_priors, 1), 1);
    
    %sorted_prod_priors(:,1) = sorted_prod_priors(:,1) - ones(size(sorted_prod_priors, 1), 1);
    %sorted_prod_priors(:,2)=  sorted_prod_priors(:,2) -  ones(size(sorted_prod_priors, 1), 1);
    
    
    %[userBelief, prodBelief, ~] = reviewZooBP(L, 0.0001, sorted_user_priors(:, 2:end), sorted_prod_priors(:, 2:end));

    user_final_beliefs = [sorted_user_priors(:, 1), userBelief(:, 1), userBelief(:, 2)];
    prod_final_beliefs = [sorted_prod_priors(:, 1), prodBelief(:, 1), prodBelief(:, 2)];
    
    csvwrite('userBeliefs.csv',user_final_beliefs); 
    csvwrite('productBeliefs.csv',prod_final_beliefs);
    fprintf('done running ZooBP\n');
    
    fid = fopen('metadata');
    C = textscan(fid, '%s %s %s %s %s');
    label = str2double(C{4});
    disp(size(label));
    user_label = zeros(size(sorted_user_priors, 1), 1);
    for i=1:size(user_priors, 1)
        user = sorted_user_priors(i, 1);
        
        user_locs = L(:, 1) == user;
        user_labels = label == -1;
        if(sum(user_locs & user_labels) > 0)
            user_label(i) = 1;
        end
    end
    csvwrite('user_labels.csv', user_label);
end