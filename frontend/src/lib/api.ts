import axios from "axios";
import { authClient } from "./auth-client";

const api = axios.create({
  baseURL: "http://localhost:8002/api",
  withCredentials: true,
});

// Request Interceptor
api.interceptors.request.use(async (config) => {
  try {
    const sessionResponse = await authClient.getSession();
    const sessionData = sessionResponse?.data as any;

    // Extract user ID directly from session (stays consistent across logins)
    const userId = sessionData?.user?.id || sessionData?.session?.userId || sessionData?.userId;

    if (userId) {
      config.headers["X-User-ID"] = userId;
      console.log("Interceptor: User ID attached:", userId);
    } else {
      console.warn("Interceptor: No user ID found in session data.");
    }

    // FastAPI trailing slash requirement fix
    if (config.url && !config.url.endsWith("/")) {
      if (!config.url.includes("?")) {
        config.url += "/";
      }
    }
  } catch (error) {
    console.error("Interceptor Error:", error);
  }

  return config;
}, (error) => {
  return Promise.reject(error);
});

export default api;