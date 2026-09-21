// Root component: login gate + sidebar + the page switcher.
// There is no router, the `page` state decides which page component is rendered
import { useEffect, useRef, useState } from "react";
import {
  BarChart2,
  Database,
  Home,
  LineChart as LineChartIcon,
  LogOut,
  Table2,
  TrendingUp,
} from "lucide-react";
import HomePage from "./pages/HomePage";
import EDAPage from "./pages/EDAPage";
import ModelsPage from "./pages/ModelsPage";
import ForecastPage from "./pages/ForecastPage";
import AdminPage from "./pages/AdminPage";
import MyUploadsPage from "./pages/MyUploadsPage";
import {
  applyDashboardPayload,
  hasDashboardData,
  ingestBackendUpload,
  resetDashboardData,
} from "./utils/dataHelpers";

// Sidebar pages. 'My Data' and 'Admin DB' are added further down depending on the user's role
const navItems = [
  { id: "home", label: "Home", Icon: Home },
  { id: "eda", label: "EDA", Icon: BarChart2 },
  { id: "models", label: "Models", Icon: TrendingUp },
  { id: "forecast", label: "Forecast", Icon: LineChartIcon },
];

// Login and register share one form, `mode` switches between them
function LoginScreen({ onLogin, onRegister, isLoading, error }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    if (mode === "register") {
      onRegister(username, password);
      return;
    }
    onLogin(username, password);
  }

  return (
    <main className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
      <section className="w-full max-w-md bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
        <div className="w-10 h-10 rounded-xl bg-slate-800 text-white flex items-center justify-center text-sm font-bold mb-5">
          FF
        </div>
        <h1 className="text-2xl font-bold text-slate-900">Financial Forecasting</h1>
        <p className="text-sm text-slate-500 mt-2">
          Sign in to upload company data and view forecasting results.
        </p>

        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <div className="grid grid-cols-2 rounded-xl border border-slate-200 overflow-hidden">
            {[
              { key: "login", label: "Sign in" },
              { key: "register", label: "Register" },
            ].map(({ key, label }) => (
              <button
                key={key}
                type="button"
                onClick={() => setMode(key)}
                className={`px-4 py-2 text-sm font-semibold ${
                  mode === key
                    ? "bg-slate-800 text-white"
                    : "bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="block">
            <span className="text-sm font-medium text-slate-700">Username</span>
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none focus:border-slate-500"
              autoComplete="username"
            />
          </label>
          <label className="block">
            <span className="text-sm font-medium text-slate-700">Password</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none focus:border-slate-500"
              autoComplete="current-password"
            />
          </label>

          {error ? <p className="text-sm text-red-600">{error}</p> : null}

          <button
            type="submit"
            disabled={isLoading || !username.trim() || !password}
            className="w-full rounded-xl bg-slate-800 px-4 py-2.5 text-sm font-semibold text-white disabled:opacity-50"
          >
            {isLoading ? "Please wait..." : mode === "register" ? "Create account" : "Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}

// The user object is cached in localStorage, so the sidebar can show the name right away
// (the token itself is still checked against the backend in the effect below)
function readStoredUser() {
  try {
    const userRaw = localStorage.getItem("ff_user");
    return userRaw ? JSON.parse(userRaw) : null;
  } catch {
    localStorage.removeItem("ff_user");
    return null;
  }
}

export default function App() {
  const [page, setPage] = useState("home");
  // The dashboard data is not React state, it lives in module variables in utils/dataHelpers.js.
  // dataVersion goes up whenever that data changes and is used as the `key` of the page below,
  // so the page is re-created and reads the new data
  const [dataVersion, setDataVersion] = useState(0);
  // The token is kept in localStorage. isChecking is true until the backend confirms that the saved token is still valid
  const [auth, setAuth] = useState(() => {
    const token = localStorage.getItem("ff_token");
    return {
      token,
      user: readStoredUser(),
      isChecking: Boolean(token),
      isLoggingIn: false,
      error: "",
    };
  });
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSummary, setUploadSummary] = useState(
    "Loading dashboard data from the backend..."
  );
  const fileInputRef = useRef(null);

  // an admin gets the uploads of all users, a normal user only their own
  async function fetchSavedUploads(token, role) {
    const endpoint = role === "admin" ? "/api/uploads/all" : "/api/uploads/my";
    const response = await fetch(endpoint, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) {
      throw new Error(`Saved uploads returned ${response.status}`);
    }
    const payload = await response.json();
    return payload.uploads ?? [];
  }

  function applySavedUploads(savedUploads) {
    resetDashboardData();
    if (!savedUploads.length) {
      setUploadSummary("Loaded default companies. Upload CSV files to add your own data.");
      setDataVersion((value) => value + 1);
      return;
    }

    savedUploads.slice().reverse().forEach((payload) => ingestBackendUpload(payload));
    setUploadSummary(
      `Loaded default companies plus ${savedUploads.length} saved upload${savedUploads.length === 1 ? "" : "s"}.`
    );
    setDataVersion((value) => value + 1);
  }

  async function reloadSavedUploads() {
    if (!auth.token) return;
    try {
      const savedUploads = await fetchSavedUploads(auth.token, auth.user?.role);
      applySavedUploads(savedUploads);
    } catch {
      resetDashboardData();
      setDataVersion((value) => value + 1);
    }
  }

  useEffect(() => {
    if (!auth.token) return undefined;

    // Runs on page load and after login:
    // 1) check the token (/api/auth/me)  2) load the default diploma dashboard  3) add the user's saved uploads on top.
    // If the backend is down we fall back to the JSON that is bundled with the frontend, so the demo still opens
    let isMounted = true;

    async function loadInitialDashboard() {
      try {
        const meResponse = await fetch("/api/auth/me", {
          headers: { Authorization: `Bearer ${auth.token}` },
        });
        if (!meResponse.ok) {
          throw new Error(`Session check returned ${meResponse.status}`);
        }

        const mePayload = await meResponse.json();
        if (!isMounted) return;
        localStorage.setItem("ff_user", JSON.stringify(mePayload.user));
        setAuth((value) => ({
          ...value,
          user: mePayload.user,
          isChecking: false,
          error: "",
        }));

        const response = await fetch("/api/dashboard/default");
        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }

        const payload = await response.json();
        if (!isMounted) return;
        const savedUploads = await fetchSavedUploads(auth.token, mePayload.user.role).catch(() => []);
        if (!isMounted) return;
        applyDashboardPayload(payload);
        if (savedUploads.length) {
          savedUploads.slice().reverse().forEach((uploadPayload) => ingestBackendUpload(uploadPayload));
          setUploadSummary(
            `Loaded default companies plus ${savedUploads.length} saved upload${savedUploads.length === 1 ? "" : "s"}.`
          );
        } else {
          setUploadSummary(
            "Loaded default companies. Upload CSV files to add your own data."
          );
        }
        setDataVersion((value) => value + 1);
      } catch (error) {
        if (!isMounted) return;
        // token rejected -> back to the login screen. Any other error (backend not running) -> use the bundled results
        if (String(error.message).includes("Session check")) {
          localStorage.removeItem("ff_token");
          localStorage.removeItem("ff_user");
          setAuth({ token: null, user: null, isChecking: false, isLoggingIn: false, error: "Session expired. Please sign in again." });
          return;
        }
        resetDashboardData();
        setUploadSummary(
          "Using bundled diploma results. Start the backend to load the default dataset from FastAPI or upload CSV files."
        );
        setDataVersion((value) => value + 1);
        setAuth((value) => ({ ...value, isChecking: false }));
      }
    }

    loadInitialDashboard();

    return () => {
      isMounted = false;
    };
  }, [auth.token]);

  async function handleLogin(username, password) {
    setAuth((value) => ({ ...value, isLoggingIn: true, error: "" }));
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        let detail = "";
        try {
          const errorPayload = await response.json();
          detail = errorPayload?.detail ? errorPayload.detail : "";
        } catch {
          detail = "";
        }
        throw new Error(detail || `Backend returned ${response.status}`);
      }

      const payload = await response.json();
      localStorage.setItem("ff_token", payload.token);
      localStorage.setItem("ff_user", JSON.stringify(payload.user));
      setAuth({
        token: payload.token,
        user: payload.user,
        isChecking: false,
        isLoggingIn: false,
        error: "",
      });
    } catch (error) {
      setAuth((value) => ({
        ...value,
        isLoggingIn: false,
        error:
          error instanceof TypeError
            ? "Backend is not reachable. Start FastAPI on port 8012."
            : `Login failed. ${error.message}`,
      }));
    }
  }

  async function handleRegister(username, password) {
    setAuth((value) => ({ ...value, isLoggingIn: true, error: "" }));
    try {
      const response = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        let detail = "";
        try {
          const errorPayload = await response.json();
          detail = errorPayload?.detail ? errorPayload.detail : "";
        } catch {
          detail = "";
        }
        throw new Error(detail || `Backend returned ${response.status}`);
      }

      const payload = await response.json();
      localStorage.setItem("ff_token", payload.token);
      localStorage.setItem("ff_user", JSON.stringify(payload.user));
      setAuth({
        token: payload.token,
        user: payload.user,
        isChecking: false,
        isLoggingIn: false,
        error: "",
      });
    } catch (error) {
      setAuth((value) => ({
        ...value,
        isLoggingIn: false,
        error:
          error instanceof TypeError
            ? "Backend is not reachable. Start FastAPI on port 8012."
            : `Registration failed. ${error.message}`,
      }));
    }
  }

  async function handleLogout() {
    if (auth.token) {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { Authorization: `Bearer ${auth.token}` },
      }).catch(() => null);
    }
    localStorage.removeItem("ff_token");
    localStorage.removeItem("ff_user");
    resetDashboardData();
    setPage("home");
    setAuth({ token: null, user: null, isChecking: false, isLoggingIn: false, error: "" });
  }

  // The files are sent one by one, the backend answers with the model results for each of them.
  // Known diploma companies come back instantly, a new company can take minutes (the pipeline is trained on the spot)
  async function handleUpload() {
    if (!selectedFiles.length || isUploading) return;

    setIsUploading(true);
    setUploadError("");
    setUploadSummary(
      `Adding ${selectedFiles.length} file${selectedFiles.length > 1 ? "s" : ""}...`
    );

    try {
      const payloads = [];

      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/api/upload?use_cache=true", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${auth.token}`,
          },
          body: formData,
        });

        if (!response.ok) {
          let detail = "";
          try {
            const errorPayload = await response.json();
            detail = errorPayload?.detail ? ` ${errorPayload.detail}` : "";
          } catch {
            detail = "";
          }
          throw new Error(`Backend returned ${response.status}.${detail}`);
        }

        payloads.push(await response.json());
      }

      const loadedCompanies = payloads.map((payload) => ingestBackendUpload(payload));

      setDataVersion((value) => value + 1);
      setUploadSummary(
        loadedCompanies.length
          ? `Added ${loadedCompanies.length} compan${loadedCompanies.length === 1 ? "y" : "ies"}.`
          : "No companies were loaded."
      );
      setSelectedFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (error) {
      setUploadError(
        error instanceof TypeError
          ? "Upload failed. Backend is not reachable at /api. Make sure FastAPI is running from /backend on port 8012."
          : `Upload failed. ${error.message}`
      );
      setUploadSummary("");
      setDataVersion((value) => value + 1);
    } finally {
      setIsUploading(false);
    }
  }

  function handleClear() {
    resetDashboardData();
    setSelectedFiles([]);
    setUploadError("");
    setUploadSummary("Dashboard reset.");
    if (fileInputRef.current) fileInputRef.current.value = "";
    setDataVersion((value) => value + 1);
  }

  // page id -> component
  const pages = {
    home: <HomePage />,
    eda: <EDAPage />,
    models: <ModelsPage />,
    forecast: <ForecastPage />,
    myData: <MyUploadsPage token={auth.token} role={auth.user?.role} onDeleted={reloadSavedUploads} />,
    admin: <AdminPage token={auth.token} />,
  };

  // admins see the uploads of everybody plus the raw database page
  const visibleNavItems = auth.user?.role === "admin"
    ? [
        ...navItems,
        { id: "myData", label: "Uploaded Data", Icon: Table2 },
        { id: "admin", label: "Admin DB", Icon: Database },
      ]
    : [...navItems, { id: "myData", label: "My Data", Icon: Table2 }];

  if (auth.isChecking) {
    return (
      <main className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
        <section className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm text-sm text-slate-600">
          Checking session...
        </section>
      </main>
    );
  }

  if (!auth.token || !auth.user) {
    return (
      <LoginScreen
        onLogin={handleLogin}
        onRegister={handleRegister}
        isLoading={auth.isLoggingIn}
        error={auth.error}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 flex">
      {/* Sidebar */}
      <aside className="w-56 bg-white border-r border-slate-100 flex flex-col py-6 px-4 fixed h-full">
        <button
          type="button"
          onClick={() => setPage("home")}
          aria-label="Go to home"
          className="mb-8 w-full text-left rounded-lg -mx-1 px-1 py-1 border-0 bg-transparent cursor-pointer transition-colors hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-300 focus-visible:ring-offset-2"
        >
          <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center mb-3">
            <span className="text-white text-xs font-bold">FF</span>
          </div>
          <h1 className="font-bold text-slate-800 text-sm leading-tight">
            Financial
            <br />
            Forecasting
          </h1>
        </button>

        <nav className="flex flex-col gap-1">
          {visibleNavItems.map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all text-left ${
                page === id
                  ? "bg-slate-800 text-white"
                  : "text-slate-600 hover:bg-slate-50"
              }`}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </nav>

        <div className="mt-auto">
          <div className="bg-slate-50 rounded-xl p-3 text-xs text-slate-500">
            <p className="font-semibold text-slate-600 mb-1">Data Modes</p>
            <p>Open the bundled diploma results by default, or replace them with CSV uploads processed by FastAPI.</p>
          </div>
          <div className="mt-3 rounded-xl border border-slate-100 p-3 text-xs text-slate-500">
            <p className="font-semibold text-slate-700">{auth.user.username}</p>
            <p className="capitalize">{auth.user.role}</p>
            <button
              type="button"
              onClick={handleLogout}
              className="mt-3 inline-flex items-center gap-2 text-sm font-medium text-slate-700 hover:text-slate-900"
            >
              <LogOut size={15} />
              Logout
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="ml-56 flex-1 p-6 lg:p-8 overflow-y-auto min-h-screen">
        <div className="max-w-[min(100%,88rem)] w-full mx-auto space-y-6">
          {page === "forecast" ? (
            <section className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
              <div className="flex flex-col lg:flex-row lg:items-end gap-4 justify-between">
                <div>
                  <p className="text-sm font-semibold text-slate-800">Data Upload</p>
                  <p className="text-sm text-slate-500 mt-1">{uploadSummary}</p>
                  {uploadError ? (
                    <p className="text-sm text-red-600 mt-2">{uploadError}</p>
                  ) : null}
                </div>

                <div className="flex flex-col sm:flex-row gap-3 sm:items-center">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    multiple
                    onChange={(event) => setSelectedFiles(Array.from(event.target.files ?? []))}
                    className="block text-sm text-slate-600 file:mr-4 file:rounded-lg file:border-0 file:bg-slate-800 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-slate-700"
                  />
                  <button
                    type="button"
                    onClick={handleUpload}
                    disabled={!selectedFiles.length || isUploading}
                    className="px-4 py-2 rounded-xl bg-slate-800 text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isUploading ? "Uploading..." : "Upload CSV"}
                  </button>
                  <button
                    type="button"
                    onClick={handleClear}
                    className="px-4 py-2 rounded-xl border border-slate-200 text-sm font-medium text-slate-600 hover:bg-slate-50"
                  >
                    Restore Bundled Data
                  </button>
                </div>
              </div>
            </section>
          ) : null}

          {page === "admin" || page === "myData" || hasDashboardData() ? (
            <div key={`${page}-${dataVersion}`}>{pages[page]}</div>
          ) : (
            <section className="bg-white border border-dashed border-slate-300 rounded-2xl p-10 text-center">
              <h2 className="text-2xl font-bold text-slate-800">Upload backend-processed company data</h2>
              <p className="text-slate-500 mt-3 max-w-2xl mx-auto">
                Choose one or more quarterly revenue CSV files, start the FastAPI backend in
                the local `backend` folder, and this dashboard will render runtime analytics
                instead of reading bundled JSON.
              </p>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
