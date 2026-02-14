library(ggplot2)
library(tidyverse)

setwd("~/Developer/ZiLU-Activation/Output/csv/LR-ResNet34")

# ZaiLU data
cifar10_test = read.csv("CIFAR10_activations_test_loss.csv")
cifar10_train = read.csv("CIFAR10_activations_train_loss.csv")

cifar100_test = read.csv("CIFAR100_activations_test_loss.csv")
cifar100_train = read.csv("CIFAR100_activations_train_loss.csv")

## [CIFAR10]
# Test Loss CIFAR10
plot_cifar10_test <- cifar10_test %>%
    pivot_longer(
        cols = -epoch,
        names_to = c("activation", "lr"),
        values_to = "loss",
        names_pattern = "(.*)_lr([^_]+)_test_loss"
    ) %>%
    filter (
      activation %in% c("relu", "silu", "gelu", "zilu_approx_sigma1.0", "zilu_sigma1.0")
    )



# Check if the pivot worked correctly

p1 <- ggplot(plot_cifar10_test, aes(x = epoch, y = loss, color = factor(activation))) +
    geom_line(linewidth = 1) +
    facet_wrap(~lr, ncol = 2) +
    labs(
        title = "CIFAR 10 ResNet34 Test Loss Curves",
        x = "Epoch",
        y = "Test Loss",
        color = "Activation"
    ) +
    theme_bw() +
    theme(legend.position = "right")
print(p1)


ggsave(
    filename = "cifar_10_test_loss.png",
    plot = p1,
    width = 8, # Width in inches
    height = 6, # Height in inches
    units = "in", # Units ("in", "cm", "mm")
    dpi = 300, # Resolution (300 is standard for print)
    bg = "white" # Background color (prevents transparent backgrounds)
)