
library(ggplot2)
library(tidyverse)
library(patchwork)

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
        str_detect(activation, "zailu|epoch"),
    ) %>%
    mutate(
        sigma = str_extract(activation, "\\d+\\.?\\d*$"),
        activation = "zailu"
    ) %>%
    mutate(
        sigma = fct_reorder(sigma, as.numeric(sigma))
    )

print(colnames(plot_gpt2_train))
print(head(plot_gpt2_train))

p1 = ggplot(plot_gpt2_train, aes(x = epoch, y = loss, color = sigma)) +
    geom_line(linewidth = 1) +
    labs(
        title = "GPT-2 WikiText103 Train Loss Curves",
        x = "Epoch",
        y = "Train Loss",
        color = "Sigma"
    ) +
    theme_bw()

ggsave(
    filename = "gpt2_zailu_train_loss.png",
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
        str_detect(activation, "zailu|epoch"),
    ) %>%
    mutate(
        sigma = str_extract(activation, "\\d+\\.?\\d*$"),
        activation = "zailu"
    ) %>%
    mutate(
        sigma = fct_reorder(sigma, as.numeric(sigma))
    )

print(colnames(plot_gpt2_val))
print(head(plot_gpt2_val))

p2 <- ggplot(plot_gpt2_val, aes(x = epoch, y = loss, color = sigma)) +
    geom_line(linewidth = 1) +
    labs(
        title = "GPT-2 WikiText103 Val Loss Curves",
        x = "Epoch",
        y = "Val Loss",
        color = "Sigma"
    ) +
    theme_bw()

ggsave(
    filename = "gpt2_zailu_val_loss.png",
    plot = p2,
    width = 8,
    height = 6,
    dpi = 300,
    units = "in",
    bg = "white"
)



# Val last 5 epochs
plot_gpt2_val_last5 <- gpt2_val %>%
    pivot_longer(
        cols = -epoch,
        names_to = "activation",
        values_to = "loss",
        names_pattern = "(.*).val"
    ) %>%
    filter(
        str_detect(activation, "zailu|epoch"),
        epoch > 15
    ) %>%
    mutate(
        sigma = str_extract(activation, "\\d+\\.?\\d*$"),
        activation = "zailu"
    ) %>%
    mutate(
        sigma = fct_reorder(sigma, as.numeric(sigma))
    )

print(colnames(plot_gpt2_val_last5))
print(head(plot_gpt2_val_last5))

p2 <- ggplot(plot_gpt2_val_last5, aes(x = epoch, y = loss, color = sigma)) +
    geom_line(linewidth = 1) +
    labs(
        title = "GPT-2 WikiText103 Val Loss Curves (Last 5 Epochs)",
        x = "Epoch",
        y = "Val Loss",
        color = "Sigma"
    ) +
    theme_bw()

ggsave(
    filename = "gpt2_zailu_val_last_5_epochs_loss.png",
    plot = p2,
    width = 8,
    height = 6,
    dpi = 300,
    units = "in",
    bg = "white"
)