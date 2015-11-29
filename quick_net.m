function quick_net(net_size, x, t)
    xn = mapminmax(x);
    
    net = newff(xn, t, net_size, {'tansig','tansig'}, 'trainlm');
    net.trainParam.epochs = 1000;
    net.trainParam.goal=0;
    
    net = train(net,xn,t);
    y = saturate(sim(net, xn));
    plotconfusion(t, y);
end