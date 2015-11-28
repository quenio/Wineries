function simple_net(net_size, x, t)
    xn = mapminmax(x);
    
    net = newff(xn, t, net_size, {'tansig','logsig'}, 'trainlm');
    net.trainParam.epochs = 1000;
    net.trainParam.goal=0.000000005;
    
    net = train(net,xn,t);
    y = saturate(sim(net, xn));
    plotconfusion(t, y);
end