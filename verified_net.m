function verified_net(net_size, x, t)
    xn = mapminmax(x);

    x_train = [xn(:,1:3:end),xn(:,3:3:end)];
    x_test = xn(:,2:3:end);
    
    t_train = [t(:,1:3:end), t(:,3:3:end)];
    t_test = t(:,2:3:end);
    
    net = newff(x_train, t_train, net_size, {'tansig','tansig'}, 'traingda');
    
    net.trainParam.epochs = 1500;
    net.trainParam.goal = 0;
    
    net.divideParam.trainRatio = 1.0;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    
    net = train(net, x_train, t_train);
    y = saturate(sim(net, x_test));
    plotconfusion(t_test, y);
end