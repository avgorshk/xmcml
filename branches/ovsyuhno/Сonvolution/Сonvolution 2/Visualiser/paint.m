
A=load('result_with_cos.txt')
imagesc(0:100, 0:125, log10(A'))
axis ([0, 100, 0, 125])


colormap(hot)
