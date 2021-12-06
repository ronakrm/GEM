library(magrittr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggthemes)
library(reshape2)
library(ggrepel)
library(scales)
library(zoo)

### Pretty Scrubbing Plots
myTheme = theme(panel.background = element_rect(fill = NA, color = "black"),
                panel.grid.major = element_line(color = "#CCCCCC"),
                panel.grid.minor = element_line(color = "#EEEEEE"),
                strip.text = element_text(size=20),
                legend.position = c(0.15,0.2),
                #legend.background = element_blank(),
                legend.box.background = element_rect(colour = "black"),
                panel.spacing = unit(2, "lines"),
                strip.background = element_blank(),
                axis.title.y = element_text(size=20),
                axis.title.x = element_text(size=20),
                axis.text.x = element_text(size=10, angle = 45, hjust = 1 ),
                axis.text.y = element_text(size=12),
                plot.title = element_text(hjust = 0.5, size=14)
)

####speed histogram
speedhistdata = data.frame(read.csv("results/speed_test_results.csv"))

speeddata = speedhistdata
#speeddata = subset(speedhistdata, gradType!='torchdual')
#speeddata = subset(speeddata, gradType!='autograd')
#speeddata = subset(speedhistdata, gradType!='npdual')
#speeddata = subset(speeddata, gradType!='scipy')
speeddata = subset(speeddata, n=10)
#speeddata = subset(speeddata, d<25)

dorder = c("2", "5", "10", "20", "50", "100")

speeddata$d <- factor(as.character(speeddata$d), levels = dorder)

gg <- ggplot(speeddata, aes(x=d, color=gradType, y=forward_time)) + 
  geom_boxplot(alpha=0.7) +
  myTheme + 
  theme(legend.position = c(0.15,0.8)) +
  ylab('Forward Time (s)') + 
  xlab('Number of Distributions') +
  guides(fill="none", alpha="none")
gg
ggsave("results/Torch_Forwards_50Bins.pdf", height = 4, width = 6, units = "in")

gg <- ggplot(speeddata, aes(x=d, color=gradType, y=backward_time)) + 
  geom_boxplot(alpha=0.7) +
  myTheme + 
  theme(legend.position = c(0.15,0.8)) +
  ylab('Backward Time (s)') + 
  xlab('Number of Distributions') +
  guides(fill="none", alpha="none")
gg
ggsave("results/Torch_Backwards_50Bins.pdf", height = 4, width = 6, units = "in")