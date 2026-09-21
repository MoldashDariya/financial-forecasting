import { useEffect, useMemo, useState } from "react";
import { Database, RefreshCw, ShieldCheck, Table2, Users } from "lucide-react";

function formatDate(value) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatRole(value) {
  return value === "admin" ? "Admin" : "User";
}

// Admin only (the backend checks the role too, hiding the button is not the protection).
// Lists users and all uploads from SQLite; click an upload to see its rows and model results
export default function AdminPage({ token }) {
  const [uploads, setUploads] = useState([]);
  const [users, setUsers] = useState([]);
  const [selectedUploadId, setSelectedUploadId] = useState(null);
  const [uploadDetail, setUploadDetail] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");

  const selectedUpload = useMemo(
    () => uploads.find((upload) => upload.id === selectedUploadId) ?? null,
    [uploads, selectedUploadId]
  );

  async function fetchAdminData() {
    setIsLoading(true);
    setError("");
    try {
      const headers = { Authorization: `Bearer ${token}` };
      const [uploadsResponse, usersResponse] = await Promise.all([
        fetch("/api/admin/uploads", { headers }),
        fetch("/api/admin/users", { headers }),
      ]);

      if (!uploadsResponse.ok) throw new Error(`Uploads returned ${uploadsResponse.status}`);
      if (!usersResponse.ok) throw new Error(`Users returned ${usersResponse.status}`);

      const uploadsPayload = await uploadsResponse.json();
      const usersPayload = await usersResponse.json();
      setUploads(uploadsPayload.uploads ?? []);
      setUsers(usersPayload.users ?? []);
      setSelectedUploadId((current) => current ?? uploadsPayload.uploads?.[0]?.id ?? null);
    } catch (requestError) {
      setError(`Could not load admin database. ${requestError.message}`);
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    fetchAdminData();
  }, [token]);

  useEffect(() => {
    if (!selectedUploadId) {
      setUploadDetail(null);
      return;
    }

    let isMounted = true;
    async function fetchDetail() {
      setDetailLoading(true);
      setError("");
      try {
        const response = await fetch(`/api/admin/uploads/${selectedUploadId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) throw new Error(`Detail returned ${response.status}`);
        const payload = await response.json();
        if (isMounted) setUploadDetail(payload);
      } catch (requestError) {
        if (isMounted) setError(`Could not load upload detail. ${requestError.message}`);
      } finally {
        if (isMounted) setDetailLoading(false);
      }
    }

    fetchDetail();
    return () => {
      isMounted = false;
    };
  }, [selectedUploadId, token]);

  const previewRows = uploadDetail?.rows?.slice(0, 12) ?? [];

  return (
    <div className="space-y-6">
      <section className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-500">
            <ShieldCheck size={16} />
            Admin-only access
          </div>
          <h2 className="text-3xl font-bold text-slate-900 mt-2">Company Database</h2>
          <p className="text-sm text-slate-500 mt-1">
            Uploaded company rows and prediction payloads stored in SQLite.
          </p>
        </div>
        <button
          type="button"
          onClick={fetchAdminData}
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

      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2 text-slate-500 text-sm font-medium">
            <Database size={16} />
            Uploads
          </div>
          <p className="text-3xl font-bold text-slate-900 mt-3">{uploads.length}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2 text-slate-500 text-sm font-medium">
            <Users size={16} />
            Users
          </div>
          <p className="text-3xl font-bold text-slate-900 mt-3">{users.length}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2 text-slate-500 text-sm font-medium">
            <Table2 size={16} />
            Rows
          </div>
          <p className="text-3xl font-bold text-slate-900 mt-3">
            {uploads.reduce((total, upload) => total + Number(upload.rows_count ?? 0), 0)}
          </p>
        </div>
      </section>

      <section className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
          <h3 className="font-semibold text-slate-900">Users</h3>
          <span className="text-xs text-slate-500">{isLoading ? "Loading..." : "SQLite table: users"}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3 text-left font-semibold">ID</th>
                <th className="px-4 py-3 text-left font-semibold">Username</th>
                <th className="px-4 py-3 text-left font-semibold">Role</th>
                <th className="px-4 py-3 text-left font-semibold">Created At</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {users.map((user) => (
                <tr key={user.id}>
                  <td className="px-4 py-3 font-medium text-slate-800">{user.id}</td>
                  <td className="px-4 py-3 text-slate-700">{user.username}</td>
                  <td className="px-4 py-3 text-slate-700">{formatRole(user.role)}</td>
                  <td className="px-4 py-3 text-slate-500">{formatDate(user.created_at)}</td>
                </tr>
              ))}
              {!users.length ? (
                <tr>
                  <td colSpan={4} className="px-4 py-8 text-center text-slate-500">
                    No users yet.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
          <h3 className="font-semibold text-slate-900">Uploads</h3>
          <span className="text-xs text-slate-500">{isLoading ? "Loading..." : "SQLite table: company_uploads"}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3 text-left font-semibold">ID</th>
                <th className="px-4 py-3 text-left font-semibold">Company</th>
                <th className="px-4 py-3 text-left font-semibold">File</th>
                <th className="px-4 py-3 text-left font-semibold">User</th>
                <th className="px-4 py-3 text-left font-semibold">Rows</th>
                <th className="px-4 py-3 text-left font-semibold">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {uploads.map((upload) => (
                <tr
                  key={upload.id}
                  onClick={() => setSelectedUploadId(upload.id)}
                  className={`cursor-pointer hover:bg-slate-50 ${
                    selectedUploadId === upload.id ? "bg-slate-50" : ""
                  }`}
                >
                  <td className="px-4 py-3 font-medium text-slate-800">{upload.id}</td>
                  <td className="px-4 py-3 text-slate-700">{upload.company_name}</td>
                  <td className="px-4 py-3 text-slate-500">{upload.filename}</td>
                  <td className="px-4 py-3 text-slate-500">{upload.username}</td>
                  <td className="px-4 py-3 text-slate-500">{upload.rows_count}</td>
                  <td className="px-4 py-3 text-slate-500">{formatDate(upload.created_at)}</td>
                </tr>
              ))}
              {!uploads.length ? (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-slate-500">
                    No company uploads yet.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100">
          <h3 className="font-semibold text-slate-900">
            {selectedUpload ? `Rows: ${selectedUpload.company_name}` : "Rows"}
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3 text-left font-semibold">Index</th>
                <th className="px-4 py-3 text-left font-semibold">Date</th>
                <th className="px-4 py-3 text-left font-semibold">Revenue</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {previewRows.map((row) => (
                <tr key={row.row_index}>
                  <td className="px-4 py-3 text-slate-500">{row.row_index}</td>
                  <td className="px-4 py-3 text-slate-700">{row.date}</td>
                  <td className="px-4 py-3 text-slate-700">{row.revenue}</td>
                </tr>
              ))}
              {!previewRows.length ? (
                <tr>
                  <td colSpan={3} className="px-4 py-8 text-center text-slate-500">
                    {detailLoading ? "Loading rows..." : "Select an upload to inspect rows."}
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
