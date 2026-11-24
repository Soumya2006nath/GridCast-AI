import axios from "axios";

const api = axios.create({ baseURL: "http://localhost:8000" });

export const loadSample = () => api.post("/api/dataset/sample");
export const uploadDataset = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/api/dataset/upload", form);
};
export const getStats = () => api.get("/api/dataset/stats");
export const getHistorical = (limit = 2000) => api.get(`/api/dataset/historical?limit=${limit}`);
export const getDaily = () => api.get("/api/dataset/daily");
export const getWeekly = () => api.get("/api/dataset/weekly");
export const getMonthly = () => api.get("/api/dataset/monthly");
export const getPreview = (rows = 100) => api.get(`/api/dataset/preview?rows=${rows}`);
export const runForecast = (model: string, lookback = 48, epochs = 30, use_cached = true) =>
  api.post("/api/forecast/run", { model, lookback, epochs, use_cached });
export const getMetrics = () => api.get("/api/metrics");
export const getHealth = () => api.get("/api/health");
export const getReportUrl = () => "http://localhost:8000/api/report/pdf";
