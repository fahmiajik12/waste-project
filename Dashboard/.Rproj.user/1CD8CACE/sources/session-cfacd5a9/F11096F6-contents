library(shiny)
library(shinydashboard)
library(readxl)
library(dplyr)
library(ggplot2)
library(plotly)
library(DT)
library(RColorBrewer)
library(lubridate)
library(scales)  # ✅ Tambahan penting

# UI
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "📊 EDA Sampah Dashboard"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem(" Ringkasan", tabName = "summary", icon = icon("chart-line")),
      menuItem(" Visualisasi", tabName = "visuals", icon = icon("chart-bar")),
      menuItem(" Tabel Data", tabName = "tabledata", icon = icon("table")),
      fileInput("file", "Upload File Excel (.xlsx)", accept = ".xlsx"),
      uiOutput("year_ui"),
      uiOutput("jenis_sampah_ui")
    )
  ),
  
  dashboardBody(
    tabItems(
      tabItem(tabName = "summary",
              fluidRow(
                valueBoxOutput("total_netto", width = 3),
                valueBoxOutput("rata_netto", width = 3),
                valueBoxOutput("max_netto", width = 3),
                valueBoxOutput("jumlah_kecamatan", width = 3),
                valueBoxOutput("jumlah_supplier", width = 3)
              ),
              fluidRow(
                box(plotlyOutput("trend_sampah", height = "700px"), width = 12, title = "Trend Bulanan Sampah", status = "primary", solidHeader = TRUE)
              )
      ),
      tabItem(tabName = "visuals",
              tabsetPanel(
                tabPanel("Total Sampah per Kecamatan",
                         fluidRow(
                           box(plotlyOutput("barplot_interaktif", height = "700px"), width = 12, title = "Total Sampah per Kecamatan", status = "info", solidHeader = TRUE)
                         )
                ),
                tabPanel("Boxplot per jenis sampah dan Supplier",
                         fluidRow(
                           box(plotlyOutput("boxplot_interaktif", height = "700px"), width = 12, title = "Boxplot per jenis sampah dan Supplier", status = "info", solidHeader = TRUE)
                         )
                ),
                tabPanel("Distribusi Sampah jenis sampah",
                         fluidRow(
                           box(plotlyOutput("barplot_jenis_sampah", height = "700px"), width = 12, title = "Distribusi Sampah jenis sampah", status = "warning", solidHeader = TRUE)
                         )
                ),
                tabPanel("Heatmap Musim & Kecamatan",
                         fluidRow(
                           box(plotOutput("heatmap", height = "800px"), width = 12, title = "Heatmap Musim & Kecamatan", status = "warning", solidHeader = TRUE)
                         )
                )
              )
      ),
      tabItem(tabName = "tabledata",
              fluidRow(
                box(DTOutput("table"), width = 12, title = "All Data", status = "success")
              )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  df <- reactive({
    req(input$file)
    data <- read_excel(input$file$datapath)
    
    required_cols <- c("Tanggal", "Hari", "Bulan", "Tahun", "No. Polisi", "jenis_sampah", "SUPPLIER", "Netto_kg", "Jam", "Sopir", "Admin", "Kecamatan", "Musim")
    validate(need(all(required_cols %in% colnames(data)), "Kolom tidak lengkap."))
    
    data <- data %>%
      mutate(
        Tahun = as.character(Tahun),
        Bulan = as.integer(Bulan),
        Kecamatan = trimws(Kecamatan),
        Musim = trimws(Musim),
        jenis_sampah = trimws(jenis_sampah),
        Tanggal = as.Date(Tanggal)
      )
    
    return(na.omit(data))
  })
  
  output$year_ui <- renderUI({
    req(df())
    selectInput("tahun", "Filter Tahun", choices = c("ALL", unique(df()$Tahun)), multiple = TRUE, selected = unique(df()$Tahun))
  })
  
  output$jenis_sampah_ui <- renderUI({
    req(df())
    selectInput("jenis_sampah", "Filter Jenis Sampah", choices = c("ALL", unique(df()$jenis_sampah)), multiple = TRUE, selected = unique(df()$jenis_sampah))
  })
  
  filtered_data <- reactive({
    data <- df()
    if (!"ALL" %in% input$tahun) data <- data %>% filter(Tahun %in% input$tahun)
    if (!"ALL" %in% input$jenis_sampah) data <- data %>% filter(jenis_sampah %in% input$jenis_sampah)
    return(data)
  })
  
  output$total_netto <- renderValueBox({
    total <- sum(filtered_data()$Netto_kg)
    valueBox(format(total, big.mark = ","), "Total Sampah (kg)", icon = icon("weight"), color = "blue")
  })
  
  output$rata_netto <- renderValueBox({
    rata <- mean(filtered_data()$Netto_kg)
    valueBox(round(rata, 2), "Rata-rata Netto (kg)", icon = icon("balance-scale"), color = "purple")
  })
  
  output$max_netto <- renderValueBox({
    maksimum <- max(filtered_data()$Netto_kg)
    valueBox(maksimum, "Netto Tertinggi (kg)", icon = icon("arrow-up"), color = "red")
  })
  
  output$jumlah_kecamatan <- renderValueBox({
    total <- length(unique(filtered_data()$Kecamatan))
    valueBox(total, "Jumlah Kecamatan", icon = icon("map-marker-alt"), color = "green")
  })
  
  output$jumlah_supplier <- renderValueBox({
    total <- length(unique(filtered_data()$SUPPLIER))
    valueBox(total, "Jumlah Supplier", icon = icon("truck-loading"), color = "teal")
  })
  
  output$trend_sampah <- renderPlotly({
    data <- filtered_data() %>%
      group_by(Tahun, Bulan) %>%
      summarise(Netto = sum(Netto_kg), .groups = 'drop')
    
    data$Bulan <- factor(data$Bulan, levels = 1:12, labels = month.name)
    
    p <- ggplot(data, aes(x = Bulan, y = Netto, color = Tahun, group = Tahun)) +
      geom_line(size = 1.5) +
      geom_point(size = 3) +
      labs(title = "Trend Bulanan Sampah", y = "Netto (kg)", x = "Bulan") +
      scale_y_continuous(labels = label_comma()) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(size = 18, face = "bold"))
    
    ggplotly(p)
  })
  
  output$barplot_interaktif <- renderPlotly({
    data <- filtered_data() %>%
      group_by(Kecamatan, Musim, Tahun) %>%
      summarise(Total = sum(Netto_kg), .groups = 'drop')
    
    p <- ggplot(data, aes(x = Kecamatan, y = Total, fill = Musim)) +
      geom_col(position = "dodge") +
      facet_wrap(~Tahun, scales = "free") +
      scale_y_continuous(labels = label_comma()) +
      theme_minimal() +
      labs(x = "Kecamatan", y = "Netto_kg", title = "") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(size = 18, face = "bold"))
    
    ggplotly(p)
  })
  
  output$boxplot_interaktif <- renderPlotly({
    p <- ggplot(filtered_data(), aes(x = jenis_sampah, y = Netto_kg, fill = SUPPLIER)) +
      geom_boxplot() +
      facet_wrap(~Tahun) +
      scale_y_continuous(labels = label_comma()) +
      labs(title = "") +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        plot.title = element_text(size = 18, face = "bold"),
        strip.text = element_text(size = 14),
        panel.spacing = unit(1, "lines")
      ) +
      scale_x_discrete(expand = c(0.1, 0.1))
    
    ggplotly(p)
  })
  
  output$heatmap <- renderPlot({
    data <- filtered_data() %>%
      group_by(Kecamatan, Musim, Tahun) %>%
      summarise(Total = sum(Netto_kg), .groups = "drop")
    
    ggplot(data, aes(x = Musim, y = reorder(Kecamatan, desc(Kecamatan)), fill = Total)) +
      geom_tile(color = "white") +
      facet_wrap(~Tahun, scales = "free_x") +
      scale_fill_gradientn(labels = label_comma(), colors = brewer.pal(9, "YlOrRd")) +
      labs(x = "Musim", y = "Kecamatan", title = "") +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 18, face = "bold"),
        strip.text = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14)
      )
  })
  
  output$barplot_jenis_sampah <- renderPlotly({
    data <- filtered_data() %>%
      group_by(jenis_sampah, Tahun) %>%
      summarise(Total = sum(Netto_kg), .groups = "drop")
    
    p <- ggplot(data, aes(x = reorder(jenis_sampah, -Total), y = Total, fill = jenis_sampah)) +
      geom_col(show.legend = FALSE) +
      facet_wrap(~Tahun) +
      scale_y_continuous(labels = label_comma()) +
      theme_minimal() +
      labs(x = "jenis_sampah", y = "Netto_kg", title = "") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(size = 18, face = "bold"))
    
    ggplotly(p)
  })
  
  output$table <- renderDT({
    datatable(filtered_data(), options = list(pageLength = 10, scrollX = TRUE))
  })
}

shinyApp(ui, server)