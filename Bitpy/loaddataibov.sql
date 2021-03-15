select
    bvp.opening as 'bvpopening', bvp.closing as 'bvpclosing',  bvp.lowest as 'bvplowest', bvp.highest as 'bvphighest',
    ptr.opening as 'ptropening', ptr.closing as 'ptrclosing',  ptr.lowest as 'ptrlowest', ptr.highest as 'ptrhighest',
    abe.opening as 'abeopening', abe.closing  as 'abeclosing', abe.lowest  as 'abelowest', abe.highest as 'abehighest',
    cmi.opening as 'cmiopening', cmi.closing  as 'cmiclosing', cmi.lowest  as 'cmilowest', cmi.highest as 'cmihighest',
    ggb.opening as 'ggbopening', ggb.closing  as 'ggbclosing', ggb.lowest  as 'ggblowest', ggb.highest as 'ggbhighest',
    itu.opening as 'ituopening', itu.closing  as 'ituclosing', itu.lowest  as 'itulowest', itu.highest as 'ituhighest',
    bbd.opening as 'bbdopening', bbd.closing  as 'bbdclosing', bbd.lowest  as 'bbdlowest', bbd.highest as 'bbdhighest',
    bba.opening as 'bbaopening', bba.closing  as 'bbaclosing', bba.lowest  as 'bbalowest', bba.highest as 'bbahighest',
    val.opening as 'valopening', val.closing  as 'valclosing', val.lowest  as 'vallowest', val.highest as 'valhighest',
    ptr.date
  from PETR4SA_day_sumary ptr,
      ABEV3SA_day_sumary abe,
      CMIG4SA_day_sumary cmi,
      GGBR4SA_day_sumary ggb,
      ITUB4SA_day_sumary itu,
      BBDC4SA_day_sumary bbd,
      BBAS3SA_day_sumary bba,
      VALE3SA_day_sumary val,
      BVSP_day_sumary bvp

 where bvp.date = ptr.date
   and abe.date = bvp.date
   and cmi.date = bvp.date
   and ggb.date = bvp.date
   and bbd.date = bvp.date
   and val.date = bvp.date
   and itu.date = bvp.date
   and bba.date = bvp.date
   and ptr.date = bvp.date

