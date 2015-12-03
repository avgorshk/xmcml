s0=load('surface0.txt')
s1=load('surface1.txt')
s2=load('surface2.txt')
A=load('new.txt')
imagesc(0:680, 0:4000, log10(A'))
axis ([0, 680, 0, 4000])


colormap(hot)
