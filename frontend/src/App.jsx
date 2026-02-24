import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Accounts from './pages/Accounts'
import AccountDetail from './pages/AccountDetail'
import ModelPerformance from './pages/ModelPerformance'
import Predict from './pages/Predict'

const NAV_ITEMS = [
    { to: '/', icon: 'dashboard', label: 'Dashboard' },
    { to: '/accounts', icon: 'group', label: 'Accounts' },
    { to: '/model', icon: 'bar_chart', label: 'Model Performance' },
    { to: '/predict', icon: 'psychology', label: 'Predict' },
]

function Sidebar() {
    const location = useLocation()

    const isActive = (to) => {
        if (to === '/') return location.pathname === '/'
        return location.pathname.startsWith(to)
    }

    return (
        <aside className="sidebar">
            <div className="sidebar-brand">
                <div className="sidebar-logo">
                    <span className="material-symbols-outlined">shield</span>
                </div>
                <div>
                    <h1>AML Guard</h1>
                    <p>Enterprise Edition</p>
                </div>
            </div>

            <nav className="sidebar-nav">
                {NAV_ITEMS.map(item => (
                    <NavLink
                        key={item.to}
                        to={item.to}
                        className={`nav-link ${isActive(item.to) ? 'active' : ''}`}
                    >
                        <span className="material-symbols-outlined">{item.icon}</span>
                        <span>{item.label}</span>
                    </NavLink>
                ))}
            </nav>
        </aside>
    )
}

export default function App() {
    return (
        <div className="app-layout">
            <Sidebar />
            <main className="main-content">
                <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/accounts" element={<Accounts />} />
                    <Route path="/accounts/:id" element={<AccountDetail />} />
                    <Route path="/model" element={<ModelPerformance />} />
                    <Route path="/predict" element={<Predict />} />
                </Routes>
            </main>
        </div>
    )
}
