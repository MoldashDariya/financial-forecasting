import { useEffect, useState } from "react";
import { RefreshCw, Trash2 } from "lucide-react";

function formatDate(value) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

// The uploads that are saved in the database. A user sees and can delete their own ones, an admin sees everybody's
export default function MyUploadsPage({ token, role, onDeleted }) {
  const [uploads, setUploads] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const isAdmin = role === "admin";
  const pageTitle = isAdmin ? "Uploaded Data" : "My Data";
  const pageDescription = isAdmin
    ? "Company uploads submitted by all users."
    : "Saved company uploads for your account.";

  async function loadUploads() {
    setIsLoading(true);
    setError("");
    try {
      const response = await fetch(isAdmin ? "/api/uploads/all" : "/api/uploads/my", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      const payload = await response.json();
      setUploads(payload.uploads ?? []);
    } catch (requestError) {
      setError(`Could not load your saved uploads. ${requestError.message}`);
    } finally {
      setIsLoading(false);
    }
  }

  async function deleteUpload(uploadId) {
    setError("");
    try {
      const response = await fetch(`/api/uploads/${uploadId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      // reload the table, then tell App to rebuild the dashboard without the deleted company
      await loadUploads();
      onDeleted();
    } catch (requestError) {
      setError(`Could not delete upload. ${requestError.message}`);
    }
  }

  useEffect(() => {
    loadUploads();
  }, [token]);

  return (
    <div className="space-y-6">
      <section className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold text-slate-900">{pageTitle}</h2>
          <p className="text-sm text-slate-500 mt-1">{pageDescription}</p>
        </div>
        <button
          type="button"
          onClick={loadUploads}
          disabled={isLoading}
          className="inline-flex items-center gap-2 rounded-xl bg-slate-800 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
        >
          <RefreshCw size={16} />
          Refresh
        </button>
      </section>

      {error ? (
        <section className="rounded-2xl border border-red-100 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </section>
      ) : null}

      <section className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100">
          <h3 className="font-semibold text-slate-900">Saved Uploads</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3 text-left font-semibold">Company</th>
                <th className="px-4 py-3 text-left font-semibold">File</th>
                <th className="px-4 py-3 text-left font-semibold">Rows</th>
                <th className="px-4 py-3 text-left font-semibold">Created</th>
                <th className="px-4 py-3 text-right font-semibold">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {uploads.map((upload) => (
                <tr key={upload.upload_id}>
                  <td className="px-4 py-3 text-slate-800 font-medium">{upload.company}</td>
                  <td className="px-4 py-3 text-slate-500">{upload.filename}</td>
                  <td className="px-4 py-3 text-slate-500">{upload.rows}</td>
                  <td className="px-4 py-3 text-slate-500">{formatDate(upload.created_at)}</td>
                  <td className="px-4 py-3 text-right">
                    <button
                      type="button"
                      onClick={() => deleteUpload(upload.upload_id)}
                      className="inline-flex items-center gap-2 rounded-lg border border-red-100 px-3 py-1.5 text-sm font-medium text-red-600 hover:bg-red-50"
                    >
                      <Trash2 size={15} />
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
              {!uploads.length ? (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-slate-500">
                    {isLoading ? "Loading..." : "No saved uploads yet."}
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
