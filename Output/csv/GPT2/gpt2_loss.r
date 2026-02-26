
library(ggplot2)
library(tidyverse)

setwd("~/Developer/ZiLU-Activation/Output/csv/GPT2/")

gpt2_train = read.csv("WikiText103_train_loss.csv")
gpt2_val = read.csv("WikiText103_val_loss.csv")

# Train Loss
plot_gpt2_train = gpt2_train %>%
    pivot_longer(
        cols = -epoch,
        names_to = "activation",
        values_to = "loss",
        names_pattern = "(.*).train"
    ) %>%
    filter(
        activation %in% c("gelu", "relu", "silu", "zailusigma5.0")
    )
            

p1 = ggplot(plot_gpt2_train, aes(x = epoch, y = loss, color = activation)) +
    geom_line(linewidth = 1) +
    labs(
        title = "GPT-2 WikiText103 Train Loss Curves",
        x = "Epoch",
        y = "Train Loss",
        color = "Activation"
    ) +
    theme_bw()

ggsave(
    filename = "gpt2_train_loss.png",
    plot = p1,
    width = 8,
    height = 6,
    dpi = 300,
    units = "in",
    bg = "white"
)

# Val Loss
plot_gpt2_val <- gpt2_val %>%
    pivot_longer(
        cols = -epoch,
        names_to = "activation",
        values_to = "loss",
        names_pattern = "(.*).val"
    ) %>%
    filter(
        activation %in% c("gelu", "relu", "silu", "zailusigma5.0")
    )


p1 = ggplot(plot_gpt2_val, aes(x = epoch, y = loss, color = activation)) +
    geom_line(linewidth = 1) +
    labs(
        title = "GPT-2 WikiText103 Val Loss Curves",
        x = "Epoch",
        y = "Val Loss",
        color = "Activation"
    ) +
    theme_bw()

ggsave(
    filename = "gpt2_val_loss.png",
    plot = p1,
    width = 8,
    height = 6,
    dpi = 300,
    units = "in",
    bg = "white"
)