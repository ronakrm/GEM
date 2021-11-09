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
speeddata = subset(speedhistdata, gradType!='npdual')
speeddata = subset(speeddata, gradType!='scipy')
speeddata = subset(speeddata, n=50)
#speeddata = subset(speeddata, d<25)

#dorder = c("2", "5", "10", "20", "50", "100")

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





##### spurious features
mydata = data.frame(read.csv("No_Beard_spur_results_agg.csv"))
levels(mydata$spurious_attr)[match("all",levels(mydata$spurious_attr))] <- "Full_Acc"
levels(mydata$spurious_attr[0])
mydata = subset(mydata, split=='val')
mydata = subset(mydata, regularizationStength==0.5 | selectionType=='None')
mydata = subset(mydata, selectionType=='FOCI')
# only for No_Beard
orderlevels = levels(mydata$spurious_attr)
orderlevels[1] = levels(mydata$spurious_attr)[2]
orderlevels[2] = levels(mydata$spurious_attr)[1]
orderlevels
mydata = mutate(mydata, spurious_attr=factor(spurious_attr, levels=orderlevels))
levels(mydata$spurious_attr)
hist <- ggplot(mydata, aes(x=spurious_attr, fill=as.character(condition), y=acc)) + 
  geom_bar(position="dodge", stat="identity", show.legend = TRUE) +
  #geom_hline(yintercept = 11*1024, linetype = "dashed", color = "blue") +
  theme(panel.background = element_rect(fill = NA, color = "black"),
        panel.grid.major = element_line(color = "#CCCCCC"),
        panel.grid.minor = element_line(color = "#EEEEEE"),
        strip.text = element_text(size=20),
        legend.position = c(0.9,0.85),
        panel.spacing = unit(2, "lines"),
        strip.background = element_blank(),
        #axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        axis.text.x = element_text(size=10, angle = 45, hjust = 1 ),
        axis.title.x = element_blank(),
        axis.text.y = element_text(size=12)
  )+ ylab('Validation Accuracy') + xlab('Selection Method') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  scale_y_continuous(limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
#scale_fill_manual(values = c("black", "blue")) + ##1b9e77", "#d95f02")) +
#scale_fill_discrete("Response")
#scale_alpha_manual(values = c(0.9,0.9))
hist
ggsave("No_Beard_FOCI.png", height = 4, width = 6, units = "in")
##### markov blanket plots
## not used in paper
mbdata = data.frame(read.csv("roc_results_test_feat.csv"))
#mbdata = subset(mbdata, conf==0.05)
mbdata = subset(mbdata, featmaps=='True')
hist <- ggplot(mbdata, aes(x=fpr, color=model, y=tpr, alpha=0.8)) + 
  geom_line(size = 2, show.legend = TRUE) +
  theme(panel.background = element_rect(fill = NA, color = "black"),
        panel.grid.major = element_line(color = "#CCCCCC"),
        panel.grid.minor = element_line(color = "#EEEEEE"),
        strip.text = element_text(size=20),
        legend.position = c(0.9,0.85),
        panel.spacing = unit(2, "lines"),
        strip.background = element_blank(),
        #axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        axis.text.x = element_text(size=10, angle = 45, hjust = 1 ),
        axis.title.x = element_text(size=20),
        axis.text.y = element_text(size=12)
  )+ ylab('True Positive Rate') + xlab('False Positive Rate') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  scale_y_continuous(limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
#scale_fill_manual(values = c("black", "blue")) + ##1b9e77", "#d95f02")) +
#scale_fill_discrete("Response")
#scale_alpha_manual(values = c(0.9,0.9))
hist
ggsave("MB_ROC.png", height = 4, width = 6, units = "in")
### MNIST
data = data.frame(read.csv("../mnist_run_Spet4/mnist_logistic.csv"), stringsAsFactors = FALSE)
data %<>%
  group_by(HessType, selectionType, n_removals) %>%
  summarise_each(funs(mean))
data = subset(data, HessType=='Sekhari')
gg <- ggplot(data, aes(x=n_removals, color=selectionType, y=val_acc_after)) + 
  geom_smooth(size = 2, show.legend = TRUE) +
  #geom_line(size=1.5, show.legend = TRUE) +
  myTheme + 
  theme(legend.position = c(0.25,0.25),
        legend.title = element_text(size=24),
        legend.text = element_text(size=20)) +
  #ggtitle('Logistic ') +
  ylab('Validation Accuracy') + 
  xlab('Removals') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  #scale_fill_manual(labels=c("Automatic", "Manual"))
  #scale_y_continuous(expand = c(0,0), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,1000))#, breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
gg
ggsave("MNIST_Valid_Acc.pdf", height = 4, width = 6, units = "in")
gg <- ggplot(data, aes(x=n_removals, color=selectionType, y=residual_acc_after)) + 
  geom_smooth(size = 2, show.legend = TRUE) +
  #geom_line(size=1.5, show.legend = TRUE) +
  myTheme + 
  theme(legend.position = c(0.25,0.25),
        legend.title = element_text(size=24),
        legend.text = element_text(size=20)) +
  ylab('Residual Accuracy') + 
  xlab('Removals') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  #scale_y_continuous(expand = c(0,0), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,1000))#, breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
gg
ggsave("MNIST_Resid_Acc.pdf", height = 4, width = 6, units = "in")
gg <- ggplot(data, aes(x=n_removals, color=selectionType, y=sample_gradnorm_after)) + 
  geom_smooth(size = 2, show.legend = TRUE) +
  #geom_line(size=1.5, show.legend = TRUE) +
  myTheme + 
  theme(legend.position = c(0.25,0.7),
        legend.title = element_text(size=24),
        legend.text = element_text(size=20)) +
  #ggtitle('Logistic ') +
  ylab('Sample Gradnorm After') + 
  xlab('Removals') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  #scale_y_continuous(expand = c(0,0), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,1000))#, breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
gg
ggsave("MNIST_GradNorm_Logistic.pdf", height = 4, width = 6, units = "in")
gg <- ggplot(data, aes(x=n_removals, color=selectionType, y=residual_gradnorm_after)) + 
  geom_smooth(size = 2, show.legend = TRUE) +
  #geom_line(size=1.5, show.legend = TRUE) +
  myTheme + 
  theme(legend.position = c(0.15,0.8)) +
  #ggtitle('Logistic ') +
  ylab('Sample Gradnorm After') + 
  xlab('Removals') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  #scale_y_continuous(expand = c(0,0), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,1000))#, breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
gg
ggsave("MNIST_GradNorm_Logistic.png", height = 4, width = 6, units = "in")
#### MNIST 2NN GRADNORM
data = data.frame(read.csv("../mnist_run_Spet4/mnist_2nn.csv"))
data %<>%
  group_by(HessType, selectionType, n_removals)# %>%
#summarise_each(funs(mean))
data = subset(data, HessType=='Sekhari')
#data = subset(data, n_removals>10)
gg <- ggplot(data, aes(x=n_removals, color=selectionType, y=sample_gradnorm_after)) + 
  geom_smooth(size = 1.5, show.legend = TRUE, method='loess') +
  #geom_line(size=1, aes(y=rollmean(sample_gradnorm_after, 100, fill=c(4,0,7.5), na.pad=TRUE))) +
  theme(panel.background = element_rect(fill = NA, color = "black"),
        panel.grid.major = element_line(color = "#CCCCCC"),
        panel.grid.minor = element_line(color = "#EEEEEE"),
        strip.text = element_text(size=20),
        legend.position = c(0.2,0.85),
        legend.background = element_blank(),
        legend.box.background = element_rect(colour = "black"),
        panel.spacing = unit(2, "lines"),
        strip.background = element_blank(),
        #axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        axis.text.x = element_text(size=10, angle = 45, hjust = 1 ),
        axis.title.x = element_text(size=20),
        axis.text.y = element_text(size=12)
  )+ ylab('Sample Gradient Norm') + xlab('Removals') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  #scale_y_continuous(expand = c(0,0), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,1000))#, breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
gg
ggsave("MNIST_GradNorm_2NN.png", height = 4, width = 6, units = "in")
### CIFAR
#cifardata = data.frame(read.csv("../cifar_Sept6_night/200_epoch_cifar_multimodel_hessiancpu_results_torchfoci.csv"))
cifardata = data.frame(read.csv("../200_epoch_cifar_multimodel_hessiancpu_results_torchfoci.csv"))
#cifardata = subset(cifardata, run==1)
cifardata %<>%
  group_by(HessType, model, n_removals) %>%
  summarise_each(funs(mean))
cifardata = subset(cifardata, HessType=='Sekhari')
cifardata = subset(cifardata, model!='vgg11_bn')
hist <- ggplot(cifardata, aes(x=n_removals, color=model, y=residual_acc_after)) + 
  #geom_smooth(size = 1.5, show.legend = TRUE, method='loess') +
  geom_line(size = 1.5, show.legend = TRUE) +
  myTheme + ylab('Residual Accuracy') + xlab('Removals') +
  theme(legend.position = c(0.2,0.3)) +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") + 
  #scale_y_continuous(expand = c(0,0.05), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(1,1000))#, trans='log10')
hist
ggsave("CIFAR_resid.pdf", height = 4, width = 6, units = "in")
hist <- ggplot(cifardata, aes(x=n_removals, color=model, y=residual_gradnorm_after)) + 
  #geom_smooth(size = 1.5, show.legend = TRUE, method='loess') +
  geom_line(size = 1.5, show.legend = TRUE) +
  myTheme + ylab('Sample Gradient Norm') + xlab('Removals') +
  theme(legend.position = c(0.8,0.25)) +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") + 
  #scale_y_continuous(expand = c(0,0.05), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(1,1000))#, trans='log10')
hist
ggsave("CIFAR_gradnom.pdf", height = 4, width = 6, units = "in")
###### LEDGAR
#### MNIST 2NN GRADNORM
data = data.frame(read.csv("../ledgar_combined.csv"))
data %<>%
  group_by(epsilon, class, n_removed)# %>%
#summarise_each(funs(mean))
data = subset(data, class==6)
data$epsilon <- as.factor(data$epsilon)
#data = subset(data, n_removals>10)
gg <- ggplot(data, aes(x=n_removed, color=epsilon, y=class_f1)) + 
  #geom_smooth(size = 1.5, show.legend = TRUE, method='loess') +
  geom_line(size=1, show.legend = TRUE) +
  myTheme + 
  ylab('Removed F1 Score') + xlab('Removals') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  #scale_y_continuous(expand = c(0,0), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,100))#, breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
gg
ggsave("LEDGAR.png", height = 4, width = 6, units = "in")
##### VGG
fname = "../vggface_scrub_Aamir_Khan_eps_0.00001.csv"
data1 = data.frame(read.csv(fname))
myd1 = data1[c('n_removals', 'resid_acc', 'scrubbed_acc_after')]
names(myd1)[2] = 'Residual Dataset'
names(myd1)[3] = 'Identity Set'
fname = "../vggface_scrub_Aamir_Khan_eps_0.00001_reps.csv"
data = data.frame(read.csv(fname))
myd = data[c('n_removals', 'residual_acc', 'scrubbed_acc_after')]
names(myd)[2] = 'Residual Dataset'
names(myd)[3] = 'Identity Set'
data = rbind(myd, myd1)
data %<>%
  group_by(n_removals) %>%
  summarise_each(funs(mean))
long <- melt(data, 'n_removals')
names(long)[2] = 'Sample_Set'
gg <- ggplot(long, aes(x=n_removals, color=Sample_Set, y=value)) + 
  #geom_smooth(size = 1.5, show.legend = TRUE, method='loess') +
  geom_line(size=1) +
  myTheme + 
  theme(legend.position = c(0.7,0.8),
        legend.title = element_blank(),
        legend.text = element_text(size=20)) +
  ylab('Accuracy') + xlab('Removals') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  #scale_y_continuous(expand = c(0,0), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,51))#, breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
gg
ggsave("VGG_Scrub_1.pdf", height = 4, width = 6, units = "in")
fname = "../vggface_scrubSI_Aamir_Khan_eps_0.00001.csv"
data = data.frame(read.csv(fname))
#myd = data[c('n_removals', 'resid_acc', 'scrubbed_acc_after')]
#names(myd)[2] = 'Residual Dataset'
#names(myd)[3] = 'Identity Set'
#long <- melt(myd, 'n_removals')
#names(long)[2] = 'Sample_Set'
gg <- ggplot(data, aes(x=n_removals, y=sample_gradnorm_change)) + 
  #geom_smooth(size = 1.5, show.legend = TRUE, method='loess') +
  geom_line(size=1) +
  myTheme + 
  theme(legend.position = c(0.8,0.8)) +
  ylab('Accuracy') + xlab('Removals') +
  guides(fill=FALSE, alpha=FALSE) + 
  scale_fill_brewer(type = "seq", palette = "Set2") +
  #scale_y_continuous(expand = c(0,0), limits = c(0,1), breaks = c(0,0.2,0.4,0.6,0.8, 1.0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,100))#, breaks = c(0,0.2,0.4,0.6,0.8, 1.0))
gg
ggsave("VGG_Scrub_1.pdf", height = 4, width = 6, units = "in")