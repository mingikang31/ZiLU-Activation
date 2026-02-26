
## ZiLU - Train + Test Loss Plots varying different alphas
library(ggplot2)
library(tidyverse)

setwd("~/Developer/ZiLU-Activation/Output")

cifar10_test = read.csv("csv/ViT-Tiny/alphas/zilu_approx/CIFAR10_test_loss.csv")
cifar10_train = read.csv("csv/ViT-Tiny/alphas/zilu_approx/CIFAR10_train_loss.csv")

cifar100_test = read.csv("csv/ViT-Tiny/alphas/zilu_approx/CIFAR100_test_loss.csv")
cifar100_train = read.csv("csv/ViT-Tiny/alphas/zilu_approx/CIFAR100_train_loss.csv")

## [CIFAR10]
# Test Loss Cifar10  
plot_cifar10_test <- cifar10_test %>%
  pivot_longer(
    cols = -epoch, 
    names_to = "sigma", 
    values_to = "loss", 
    names_pattern = "zilu_approx_sigma(.*)_test_loss"
  ) %>%
  mutate(sigma = as.numeric(sigma)) 

p1 = ggplot(plot_cifar10_test, aes(x = epoch, y = loss, color = factor(sigma))) +
  geom_line(linewidth = 1) +
  labs(
    title = "CIFAR 10 ZaiLU Approx Test Loss Curves",
    x = "Epoch",
    y = "Test Loss",
    color = "Sigma"
  ) +
  theme_bw()

print(p1)

# train Loss Cifar10  
plot_cifar10_train <- cifar10_train %>%
  pivot_longer(
    cols = -epoch, 
    names_to = "sigma", 
    values_to = "loss", 
    names_pattern = "zilu_approx_sigma(.*)_train_loss"
  ) %>%
  mutate(sigma = as.numeric(sigma)) 

p2 = ggplot(plot_cifar10_train, aes(x = epoch, y = loss, color = factor(sigma))) +
  geom_line(linewidth = 1) +
  labs(
    title = "CIFAR 10 ZaiLU Approx Train Loss Curves",
    x = "Epoch",
    y = "Train Loss",
    color = "Sigma"
  ) +
  theme_bw()

print(p2)

## [CIFAR100]

# Test Loss  
plot_cifar100_test <- cifar100_test %>%
  pivot_longer(
    cols = -epoch, 
    names_to = "sigma", 
    values_to = "loss", 
    names_pattern = "zilu_approx_sigma(.*)_test_loss"
  ) %>%
  mutate(sigma = as.numeric(sigma)) 

p3 = ggplot(plot_cifar100_test, aes(x = epoch, y = loss, color = factor(sigma))) +
  geom_line(linewidth = 1) +
  labs(
    title = "CIFAR 100 ZaiLU Approx Test Loss Curves",
    x = "Epoch",
    y = "Test Loss",
    color = "Sigma"
  ) +
  theme_bw()

print(p3)

# train Loss
plot_cifar100_train <- cifar100_train %>%
  pivot_longer(
    cols = -epoch,
    names_to = "sigma",
    values_to = "loss",
    names_pattern = "zilu_approx_sigma(.*)_train_loss"
  ) %>%
  mutate(sigma = as.numeric(sigma))

p4 = ggplot(plot_cifar100_train, aes(x = epoch, y = loss, color = factor(sigma))) +
  geom_line(linewidth = 1) +
  labs(
    title = "CIFAR 100 ZaiLU Approx train Loss Curves",
    x = "Epoch",
    y = "Train Loss",
    color = "Sigma"
  ) +
  theme_bw()

print(p4)

ggsave(
  filename = "csv/ViT-Tiny/alphas/zilu_approx/cifar10_sigma_test_loss.png",
  plot = p1,
  width = 8,       # Width in inches
  height = 6,      # Height in inches
  units = "in",    # Units ("in", "cm", "mm")
  dpi = 300,       # Resolution (300 is standard for print)
  bg = "white"     # Background color (prevents transparent backgrounds)
)

ggsave(
  filename = "csv/ViT-Tiny/alphas/zilu_approx/cifar10_sigma_train_loss.png",
  plot = p2,
  width = 8,       # Width in inches
  height = 6,      # Height in inches
  units = "in",    # Units ("in", "cm", "mm")
  dpi = 300,       # Resolution (300 is standard for print)
  bg = "white"     # Background color (prevents transparent backgrounds)
)

ggsave(
  filename = "csv/ViT-Tiny/alphas/zilu_approx/cifar100_sigma_test_loss.png",
  plot = p3,
  width = 8,       # Width in inches
  height = 6,      # Height in inches
  units = "in",    # Units ("in", "cm", "mm")
  dpi = 300,       # Resolution (300 is standard for print)
  bg = "white"     # Background color (prevents transparent backgrounds)
)

ggsave(
  filename = "csv/ViT-Tiny/alphas/zilu_approx/cifar100_sigma_train_loss.png",
  plot = p4,
  width = 8,       # Width in inches
  height = 6,      # Height in inches
  units = "in",    # Units ("in", "cm", "mm")
  dpi = 300,       # Resolution (300 is standard for print)
  bg = "white"     # Background color (prevents transparent backgrounds)
)
