library(ggplot2)
library(tidyverse)
library(zoo)

setwd("~/Developer/ZiLU-Activation/Output/csv/LR-ResNet34")

# ZaiLU data
cifar10_test = read.csv("CIFAR10_activations_test_loss.csv")
cifar10_train = read.csv("CIFAR10_activations_train_loss.csv")

cifar100_test = read.csv("CIFAR100_activations_test_loss.csv")
cifar100_train = read.csv("CIFAR100_activations_train_loss.csv")


## cifar_100_test_loss.png code
# cifar_100_test <- cifar100_test %>%
#     pivot_longer(
#         cols = -epoch,
#         names_to = c("activation", "lr"),
#         values_to = "loss",
#         names_pattern = "^(.+?)\\.lr(.+?)\\.test$"
#     ) %>%
#     filter(
#         activation %in% c("relu", "silu", "gelu", "zailuapproxsigma1.0", "zailusigma1.0")
#     )

# print(head(cifar_100_test))
# print(unique(cifar_100_test$activation))


# p1 = ggplot(cifar_100_test, aes(x = epoch, y = loss, color = activation)) +
#     geom_line(linewidth = 1) +
#     facet_wrap(~lr, ncol = 2, scales = "free_y") +
#     labs(
#         title = "CIFAR 100 ResNet34 Test Loss Curves",
#         x = "Epoch",
#         y = "Test Loss",
#         color = "Activation"
#     ) +
#     theme_bw() +
#     theme(legend.position = "right")

# ggsave(
#     filename = "cifar_100_test_loss.png",
#     plot = p1,
#     width = 8, # Width in inches
#     height = 6, # Height in inches
#     units = "in", # Units ("in", "cm", "mm")
#     dpi = 300, # Resolution (300 is standard for print)
#     bg = "white" # Background color (prevents transparent backgrounds)
# )


# Doing LR = 1e-4
cifar_100_test <- cifar100_test %>%
    pivot_longer(
        cols = -epoch,
        names_to = c("activation", "lr"),
        values_to = "loss",
        names_pattern = "^(.+?)\\.lr(.+?)\\.test$"
    ) %>%
    filter(
        activation %in% c("relu", "silu", "gelu", "zailusigma0.1", "zailusigma0.5", "zailusigma1.0", "zailusigma5.0", "zailusigma10.0"),
        lr == "1e.4"
    ) %>%
    mutate(
        loss = rollmean(loss, k = 10, fill = NA)
    ) %>%
    filter (
        epoch > 100
    )

print(head(cifar_100_test))
print(unique(cifar_100_test$activation))


p1 <- ggplot(cifar_100_test, aes(x = epoch, y = loss, color = activation)) +
    geom_line(linewidth = 1) +
    labs(
        title = "CIFAR 100 ResNet34 Test Loss Curves",
        x = "Epoch",
        y = "Test Loss",
        color = "Activation"
    ) +
    theme_bw() +
    theme(legend.position = "right")

ggsave(
    filename = "cifar_100_lr1e-4_test_loss.png",
    plot = p1,
    width = 8, # Width in inches
    height = 10, # Height in inches
    units = "in", # Units ("in", "cm", "mm")
    dpi = 300, # Resolution (300 is standard for print)
    bg = "white" # Background color (prevents transparent backgrounds)
)


