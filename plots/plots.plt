#!/usr/bin/env gnuplot
reset
WIDTH  = 4.2
HEIGHT = 2.4
set terminal postscript eps size WIDTH,HEIGHT enhanced color font 'Helvetica,16'
set output "forward_pass_alexnet.eps"

#set style data histogram
#set style histogram cluster gap 1
#set style fill solid border -1
#set boxwidth 0.8
set logscale y
#set yrange[0.0001:0.01]
set xlabel "Number of Layer"
set ylabel "Time (s)"
#set y2label "MassiveThreads Total Execution Time (ms)"

set grid ytics lw 1 lt 0
set xtics scale 0
set key center top horizontal Left reverse noenhanced autotitle box





filename(n) = sprintf("alexnet_static_%d.dat", n)
filename2(n) = sprintf("vgg16_static_%d.dat", n)
titlename(n) = sprintf("%d procs", n)
sfilename(n) = sprintf("alexnet_%s.dat", n)
stitlename(n) = sprintf("%s", n)
filenames = "8 9 11 12 15 16"
#plot for [i in filenames] sfilename(i) using 1:5 with lp ti stitlename(i)

set output "forward_pass_alexnet.eps"
plot for [i=2:16:2] filename(i) using 1:2 with lp ti titlename(i)


set output "backward_pass_alexnet.eps"
plot for [i=2:16:2] filename(i) using 1:5 with lp ti titlename(i)

set yrange[0.09:5]
set output "forward_pass_vgg16.eps"
plot for [i=2:16:2] filename2(i) using 1:2 with lp ti titlename(i)

set autoscale y
set output "backward_pass_vgg16.eps"
plot for [i=2:16:2] filename2(i) using 1:5 with lp ti titlename(i)
