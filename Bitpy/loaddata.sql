select
    btc.opening as 'btcopening', btc.closing as 'btcclosing', btc.quantity as 'btcquantity', btc.amount as 'btcamount',
    dax.opening as 'daxopening', dax.closing  as 'daxclosing', dax.lowest  as 'daxlowest', dax.highest as 'daxhighest',
    dji.opening as 'djiopening', dji.closing  as 'djiclosing', dji.lowest  as 'djilowest', dji.highest as 'djihighest',
    hsi.opening as 'hsiopening', hsi.closing  as 'hsiclosing', hsi.lowest  as 'hsilowest', hsi.highest as 'hsihighest',
    btc.date
 from BTC_day_sumary btc, DAX_day_sumary dax, DJI_day_sumary dji, HSI_day_sumary hsi
 where btc.date = dax.date
 and   dax.date = dji.date
 and   dji.date = hsi.date;

